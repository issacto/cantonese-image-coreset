"""
push_to_hub.py
──────────────
Merge LoRA weights into the base LLM, package with the VisionProjection,
and push everything to the Hugging Face Hub as a single repository.

Usage
─────
    python push_to_hub.py \
        --projector_path  ./output/projector_final.pt \
        --lora_path       ./output/lora_adapter_final \
        --llm_model_id    meta-llama/Llama-3.2-1B \
        --repo_id         your-hf-username/my-vision-llm \
        --dtype           bf16

    # Optional flags
        --private                     # make the repo private
        --commit_message "v1 release" # custom commit message

What gets pushed
────────────────
  <repo_id>/
  ├── projector/
  │   └── projector.pt          # merged projector weights + config dict
  ├── llm/                      # full merged (non-PEFT) LLM weights
  │   ├── config.json
  │   ├── tokenizer*.json / *.model
  │   └── model-*.safetensors
  └── README.md                 # auto-generated model card
"""

import argparse
import os
import tempfile
import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_dtype(dtype_str: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]


def load_and_merge(projector_path: str, lora_path: str, llm_model_id: str,
                   dtype: torch.dtype, device: str):
    """
    1. Load the projector checkpoint (state_dict + config).
    2. Load base LLM → apply LoRA → merge & unload LoRA → plain nn.Module.
    3. Load tokenizer.
    Returns (projector_ckpt, merged_llm, tokenizer).
    """
    print("Loading projector …")
    projector_ckpt = torch.load(projector_path, map_location=device)
    # Supports both old format (bare state_dict) and new format (dict with "config")
    if "state_dict" not in projector_ckpt:
        projector_ckpt = {"state_dict": projector_ckpt, "config": {}}

    print(f"Loading base LLM: {llm_model_id} …")
    base_llm = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=dtype, device_map=device
    )

    print(f"Applying LoRA from: {lora_path} …")
    peft_model = PeftModel.from_pretrained(base_llm, lora_path)

    print("Merging LoRA weights into base model …")
    merged_llm = peft_model.merge_and_unload()   # returns a plain HF model
    merged_llm.eval()

    tokenizer = AutoTokenizer.from_pretrained(llm_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return projector_ckpt, merged_llm, tokenizer


def build_model_card(repo_id: str, llm_model_id: str, projector_config: dict) -> str:
    clip_dim = projector_config.get("clip_dim", "?")
    llm_dim  = projector_config.get("llm_dim",  "?")
    return textwrap.dedent(f"""\
        ---
        license: apache-2.0
        tags:
          - vision-language
          - multimodal
          - clip
          - lora-merged
        ---

        # {repo_id.split("/")[-1]}

        Vision-language model trained with a CLIP vision encoder, a learned
        VisionProjection MLP, and a LoRA-fine-tuned causal LM backbone.

        ## Architecture

        | Component | Detail |
        |-----------|--------|
        | Base LLM  | `{llm_model_id}` (LoRA merged) |
        | CLIP dim  | {clip_dim} |
        | LLM dim   | {llm_dim} |

        ## Files

        | Path | Contents |
        |------|----------|
        | `projector/projector.pt` | VisionProjection weights + config |
        | `llm/` | Merged LLM (SafeTensors) + tokenizer |

        ## Loading

        ```python
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import hf_hub_download
        from core.models import VisionProjection

        REPO = "{repo_id}"

        # Projector
        ckpt = torch.load(hf_hub_download(REPO, "projector/projector.pt"))
        cfg  = ckpt["config"]
        projector = VisionProjection(cfg["clip_dim"], cfg["llm_dim"])
        projector.load_state_dict(ckpt["state_dict"])
        projector.eval()

        # LLM + tokenizer
        llm = AutoModelForCausalLM.from_pretrained(f"{{REPO}}/llm")
        tokenizer = AutoTokenizer.from_pretrained(f"{{REPO}}/llm")
        ```
    """)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Push vision-LLM to Hugging Face Hub")
    parser.add_argument("--projector_path",  required=True,
                        help="Path to projector_final.pt (or any step checkpoint)")
    parser.add_argument("--lora_path",       required=True,
                        help="Directory containing the saved LoRA adapter")
    parser.add_argument("--llm_model_id",    required=True,
                        help="HuggingFace model ID or local path for the base LLM")
    parser.add_argument("--repo_id",         required=True,
                        help="Hub repo to push to, e.g. 'username/my-vision-llm'")
    parser.add_argument("--dtype",           default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device",          default="cuda",
                        help="Device for loading ('cuda', 'cpu', 'cuda:0', …)")
    parser.add_argument("--private",         action="store_true",
                        help="Create a private Hub repository")
    parser.add_argument("--commit_message",  default="Upload merged vision-LLM")
    args = parser.parse_args()

    dtype = resolve_dtype(args.dtype)

    # ── 1. Load & merge ───────────────────────────────────────────────────────
    projector_ckpt, merged_llm, tokenizer = load_and_merge(
        args.projector_path, args.lora_path, args.llm_model_id, dtype, args.device
    )

    api = HfApi()

    # ── 2. Ensure the repo exists ─────────────────────────────────────────────
    print(f"\nEnsuring Hub repo '{args.repo_id}' exists …")
    create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")

    # ── 3. Stage everything in a temp dir, then push ──────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:

        # 3a. Projector
        proj_dir = os.path.join(tmpdir, "projector")
        os.makedirs(proj_dir)
        proj_out  = os.path.join(proj_dir, "projector.pt")
        torch.save(projector_ckpt, proj_out)
        print(f"  Saved projector → {proj_out}")

        # 3b. Merged LLM + tokenizer
        llm_dir = os.path.join(tmpdir, "llm")
        os.makedirs(llm_dir)
        print(f"  Saving merged LLM to {llm_dir} …")
        merged_llm.save_pretrained(llm_dir, safe_serialization=True)
        tokenizer.save_pretrained(llm_dir)

        # 3c. Model card
        readme_path = os.path.join(tmpdir, "README.md")
        with open(readme_path, "w") as f:
            f.write(build_model_card(
                args.repo_id,
                args.llm_model_id,
                projector_ckpt.get("config", {}),
            ))

        # 3d. Upload the whole staged directory
        print(f"\nUploading to https://huggingface.co/{args.repo_id} …")
        api.upload_folder(
            folder_path=tmpdir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

    print(f"\n✓ Done!  https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()