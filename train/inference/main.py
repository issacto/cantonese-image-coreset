"""
inference.py
────────────
Run inference with your pushed vision-LLM.

Usage (in a notebook cell):
    from inference import VisionLLM
    model = VisionLLM("Issactoto/granite-4.1-3b-vl", clip_model_id="openai/clip-vit-large-patch14")
    response = model.chat("Describe this image.", image="path/to/image.jpg")
    print(response)
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor
from huggingface_hub import hf_hub_download
import requests
from io import BytesIO


# ── VisionProjection (must match your training definition) ────────────────────

import torch.nn as nn

class VisionProjection(nn.Module):
    """Two-layer MLP that projects CLIP patch embeddings → LLM token space."""
    def __init__(self, clip_dim: int, llm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
    def forward(self, x):
        return self.net(x)


# ── Main inference class ──────────────────────────────────────────────────────

class VisionLLM:
    def __init__(
        self,
        repo_id: str = "Issactoto/granite-4.1-3b-vl",
        clip_model_id: str = "openai/clip-vit-large-patch14",
        dtype: str = "bf16",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype  = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

        print(f"Device: {self.device}  |  dtype: {dtype}")

        # ── CLIP ──────────────────────────────────────────────────────────────
        print(f"Loading CLIP: {clip_model_id} …")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(
            clip_model_id, torch_dtype=self.dtype
        ).to(self.device)
        self.clip_model.eval()

        # ── Projector ─────────────────────────────────────────────────────────
        print(f"Loading projector from {repo_id} …")
        proj_path = hf_hub_download(repo_id, "projector/projector.pt")
        ckpt = torch.load(proj_path, map_location=self.device)

        # Handle both formats: final (with config) and step (bare state_dict)
        if "config" in ckpt and ckpt["config"]:
            cfg       = ckpt["config"]
            clip_dim  = cfg["clip_dim"]
            llm_dim   = cfg["llm_dim"]
            state_dict = ckpt["state_dict"]
        else:
            # Infer dims from weight shapes
            state_dict = ckpt.get("state_dict", ckpt)
            first_w = next(v for k, v in state_dict.items() if "weight" in k)
            last_w  = [v for k, v in state_dict.items() if "weight" in k][-1]
            clip_dim = first_w.shape[1]
            llm_dim  = last_w.shape[0]
            print(f"  Inferred dims — CLIP: {clip_dim}  LLM: {llm_dim}")

        self.projector = VisionProjection(clip_dim, llm_dim)
        self.projector.load_state_dict(state_dict)
        self.projector.eval().to(self.device, dtype=self.dtype)

        # ── LLM + tokenizer ───────────────────────────────────────────────────
        print(f"Loading LLM from {repo_id}/llm …")
        llm_path = f"{repo_id}/llm" if "/" in repo_id else repo_id
        self.tokenizer = AutoTokenizer.from_pretrained(f"{repo_id}/llm" if not repo_id.endswith("/llm") else repo_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            f"{repo_id}/llm",
            torch_dtype=self.dtype,
            device_map=str(self.device),
        )
        self.llm.eval()
        print("✓ Model ready!\n")

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self, image) -> Image.Image:
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                response = requests.get(image, timeout=10)
                return Image.open(BytesIO(response.content)).convert("RGB")
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise ValueError("image must be a file path, URL, or PIL.Image")

    # ── Core forward ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_image(self, image) -> torch.Tensor:
        """Returns visual tokens of shape (1, P, llm_dim)."""
        pil = self._load_image(image)
        inputs = self.clip_processor(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        vision_out    = self.clip_model.vision_model(pixel_values=pixel_values)
        patch_tokens  = vision_out.last_hidden_state[:, 1:, :]  # drop CLS
        visual_tokens = self.projector(patch_tokens)             # (1, P, llm_dim)
        return visual_tokens

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def chat(
        self,
        prompt: str,
        image=None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response.

        Parameters
        ----------
        prompt        : Text prompt / question.
        image         : File path, URL, or PIL.Image. Pass None for text-only.
        max_new_tokens: Maximum tokens to generate.
        temperature   : Sampling temperature (ignored if do_sample=False).
        do_sample     : Use sampling (True) or greedy (False).
        """
        tok = self.tokenizer(prompt, return_tensors="pt")
        input_ids      = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)

        # Text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (1, T, llm_dim)

        if image is not None:
            visual_tokens = self._encode_image(image)             # (1, P, llm_dim)
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            # Extend attention mask for visual prefix
            vis_mask      = torch.ones(1, visual_tokens.shape[1],
                                       dtype=attention_mask.dtype, device=self.device)
            full_mask     = torch.cat([vis_mask, attention_mask], dim=1)
        else:
            inputs_embeds = text_embeds
            full_mask     = attention_mask

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        generated = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = VisionLLM(
        repo_id="Issactoto/granite-4.1-3b-vl",
        clip_model_id="openai/clip-vit-large-patch14",
    )

    # Test with a URL image
    print("=== Image + prompt ===")
    reply = model.chat(
        prompt="Describe what you see in this image.",
        image="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg",
    )
    print(reply)

    # Text-only test
    print("\n=== Text only ===")
    reply = model.chat(prompt="What is the capital of France?", image=None)
    print(reply)