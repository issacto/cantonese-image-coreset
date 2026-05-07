"""
Multi-node, multi-GPU vision-LLM training with Ray Train.

Trainable components
────────────────────
  • VisionProjection  – MLP bridge between CLIP and the LLM token space
  • LoRA adapters     – injected into the attention layers of the LLM

Frozen components
─────────────────
  • CLIP vision encoder
  • All non-LoRA LLM parameters

Parallelism strategies
──────────────────────
  ddp   – each worker holds a full model replica; gradients are all-reduced.
           Best when the LLM fits on a single GPU.
  fsdp  – shards model parameters, gradients and optimiser state across GPUs
           via PyTorch FSDP. Use when the LLM is too large for one GPU.

Usage
─────
  # Minimal – single node, all local GPUs, DDP:
  python train_ray_lora.py \
      --clip_model openai/clip-vit-base-patch32 \
      --llm_model  hon9kon9ize/CantoneseLLMChat-v1.0-7B \
      --dataset    Issactoto/flickr8k-cantonese \
      --image_col  image \
      --text_col   yue_caption

  # Multi-node FSDP with explicit resources:
  RAY_ADDRESS=ray://<head>:10001 python train_ray_lora.py \
      --clip_model       openai/clip-vit-large-patch14 \
      --llm_model        meta-llama/Llama-3-8b-hf \
      --dataset          nlphuji/flickr30k \
      --image_col        image \
      --text_col         caption \
      --parallelism      fsdp \
      --num_workers      8 \
      --gpus_per_worker  1 \
      --cpus_per_worker  4 \
      --batch_size       2 \
      --lora_r           32 \
      --epochs           5 \
      --output_dir       ./my_checkpoints

Dependencies
────────────
  pip install "ray[train]>=2.10" torch transformers datasets peft tqdm pillow
"""

import argparse
import os
from pathlib import Path

import ray
import ray.train
from ray.train import ScalingConfig, CheckpointConfig, RunConfig
from ray.train.torch import TorchTrainer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPModel,
    CLIPProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import load_dataset
from tqdm import tqdm


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a vision-LLM projection layer + LoRA adapters with Ray.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    m = p.add_argument_group("Models")
    m.add_argument("--clip_model", required=True,
                   help="HuggingFace CLIP model ID or local path.")
    m.add_argument("--llm_model",  required=True,
                   help="HuggingFace causal-LM model ID or local path.")
    m.add_argument("--clip_dim",   type=int, default=None,
                   help="CLIP pooled output dim. Auto-detected from model config if omitted.")
    m.add_argument("--llm_dim",    type=int, default=None,
                   help="LLM hidden dim. Auto-detected from model config if omitted.")
    m.add_argument("--dtype",      choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="Parameter and autocast dtype.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    d = p.add_argument_group("Dataset")
    d.add_argument("--dataset",        required=True,
                   help="HuggingFace dataset repo ID or local path.")
    d.add_argument("--dataset_split",  default="train",
                   help="Split to use, e.g. 'train' or 'train[:80%%]'.")
    d.add_argument("--dataset_config", default=None,
                   help="Optional dataset config/subset name.")
    d.add_argument("--image_col",      required=True,
                   help="Column that contains PIL images or file-path strings.")
    d.add_argument("--text_col",       required=True,
                   help="Column that contains caption / text target strings. "
                        "If a row holds a list of captions the first is used.")
    d.add_argument("--max_text_len",   type=int, default=128,
                   help="Maximum tokenised text length (tokens).")

    # ── Training hyper-params ─────────────────────────────────────────────────
    t = p.add_argument_group("Training")
    t.add_argument("--batch_size",    type=int,   default=4,
                   help="Per-worker micro-batch size.")
    t.add_argument("--grad_accum",    type=int,   default=4,
                   help="Gradient accumulation steps.")
    t.add_argument("--epochs",        type=int,   default=3)
    t.add_argument("--lr",            type=float, default=2e-4)
    t.add_argument("--warmup_ratio",  type=float, default=0.03)
    t.add_argument("--weight_decay",  type=float, default=0.01)
    t.add_argument("--max_grad_norm", type=float, default=1.0)
    t.add_argument("--save_every",    type=int,   default=500,
                   help="Save a step checkpoint every N global optimiser steps.")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    l = p.add_argument_group("LoRA")
    l.add_argument("--lora_r",       type=int,   default=16)
    l.add_argument("--lora_alpha",   type=int,   default=32)
    l.add_argument("--lora_dropout", type=float, default=0.05)
    l.add_argument("--lora_targets", nargs="+",  default=["q_proj", "v_proj"],
                   help="LLM module names to inject LoRA adapters into.")

    # ── Ray / cluster resources ───────────────────────────────────────────────
    r = p.add_argument_group("Ray / cluster")
    r.add_argument("--parallelism", choices=["ddp", "fsdp"], default="ddp",
                   help=(
                       "ddp  – full model replica per GPU; gradients all-reduced. "
                       "       Best when the LLM fits on a single GPU. "
                       "fsdp – shards model + optimiser state across all GPUs. "
                       "       Required for LLMs that exceed single-GPU VRAM."
                   ))
    r.add_argument("--num_workers",     type=int,   default=1,
                   help="Total Ray workers (= total GPUs) across all nodes.")
    r.add_argument("--gpus_per_worker", type=float, default=1.0,
                   help="GPU fraction per worker (use 1.0 for dedicated GPUs).")
    r.add_argument("--cpus_per_worker", type=int,   default=2,
                   help="CPUs per worker; also used as DataLoader num_workers.")
    r.add_argument("--ray_address",     default=None,
                   help="Ray cluster address (e.g. ray://<head>:10001). "
                        "Omit to start a local cluster automatically.")

    # ── Output ────────────────────────────────────────────────────────────────
    o = p.add_argument_group("Output")
    o.add_argument("--output_dir",           default="./ray_lora_checkpoints")
    o.add_argument("--run_name",             default="vision_lora_train",
                   help="Label shown in Ray dashboard / results folder.")
    o.add_argument("--checkpoints_to_keep",  type=int, default=3,
                   help="How many Ray trial checkpoints to retain.")

    return p.parse_args(argv)


# ─── Vision Projection ────────────────────────────────────────────────────────

class VisionProjection(nn.Module):
    """Two-layer GELU MLP: CLIP pooled dim → LLM hidden dim."""

    def __init__(self, clip_dim: int, llm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Generic vision-text dataset ──────────────────────────────────────────────

class VisionTextDataset(Dataset):
    """
    General-purpose wrapper for any HuggingFace image-caption dataset.

    Supports:
      • PIL Image objects stored directly in the dataset column.
      • File-path strings that will be opened with Pillow.
      • Caption columns that contain a single string or a list of strings
        (first caption is used by default; override with `text_selector`).

    Parameters
    ----------
    hf_dataset     : HuggingFace Dataset object (already loaded and split).
    clip_processor : CLIPProcessor for image pre-processing.
    tokenizer      : Any HuggingFace tokenizer.
    image_col      : Column name for images (PIL or path string).
    text_col       : Column name for captions.
    max_len        : Maximum tokenised sequence length.
    text_selector  : Optional callable(raw_value) -> str. Use this to pick
                     one caption from a list or to apply any custom logic.
                     Defaults to: first element if list, else the value as-is.
    """

    def __init__(
        self,
        hf_dataset,
        clip_processor,
        tokenizer,
        image_col: str,
        text_col: str,
        max_len: int = 128,
        text_selector=None,
    ):
        self.data          = hf_dataset
        self.clip_proc     = clip_processor
        self.tokenizer     = tokenizer
        self.image_col     = image_col
        self.text_col      = text_col
        self.max_len       = max_len
        self.text_selector = text_selector or (
            lambda v: v[0] if isinstance(v, (list, tuple)) else v
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        # ── Image ──────────────────────────────────────────────────────────────
        raw = item[self.image_col]
        if isinstance(raw, str):
            from PIL import Image
            raw = Image.open(raw)
        image = raw.convert("RGB")

        pixel_values = self.clip_proc(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # ── Text ───────────────────────────────────────────────────────────────
        caption  = self.text_selector(item[self.text_col])
        encoding = self.tokenizer(
            caption,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "pixel_values":   pixel_values,
            "input_ids":      encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
        }


# ─── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(
    batch: dict,
    clip_model: nn.Module,
    projector: nn.Module,
    llm: nn.Module,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Prepend the projected visual token to the caption token embeddings,
    then compute next-token prediction loss. The visual-token label is
    masked with -100 so the model is only supervised on the caption.
    """
    pixel_values   = batch["pixel_values"].to(device, dtype=dtype)
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # CLIP (frozen)
    with torch.no_grad():
        image_features = clip_model.vision_model(
            pixel_values=pixel_values
        ).pooler_output.to(dtype)

    visual_tokens = projector(image_features).unsqueeze(1)   # (B, 1, D)

    # Reach the raw embed_tokens table through DDP / FSDP / PEFT wrapping
    inner = llm
    for attr in ("module", "base_model", "model"):
        inner = getattr(inner, attr, inner)

    with torch.no_grad():
        text_embeds = inner.embed_tokens(input_ids)

    inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

    visual_mask = torch.ones(
        pixel_values.size(0), 1, device=device, dtype=attention_mask.dtype
    )
    full_mask = torch.cat([visual_mask, attention_mask], dim=1)

    visual_labels = torch.full(
        (pixel_values.size(0), 1), -100, dtype=torch.long, device=device
    )
    labels = torch.cat([visual_labels, input_ids], dim=1)
    labels[labels == tokenizer.pad_token_id] = -100

    return llm(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels,
    ).loss


# ─── Small helpers ────────────────────────────────────────────────────────────

def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def unwrap(model: nn.Module) -> nn.Module:
    """Strip one layer of DDP / FSDP wrapping."""
    return model.module if hasattr(model, "module") else model


def autodetect_clip_dim(clip_model: CLIPModel) -> int:
    return clip_model.config.vision_config.hidden_size


def autodetect_llm_dim(llm) -> int:
    cfg = unwrap(llm).config
    return getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))


# ─── Checkpoint I/O ───────────────────────────────────────────────────────────

def _save_checkpoint(projector, llm, output_dir: str, suffix: str):
    torch.save(
        unwrap(projector).state_dict(),
        os.path.join(output_dir, f"projector_{suffix}.pt"),
    )
    unwrap(llm).save_pretrained(os.path.join(output_dir, f"lora_adapter_{suffix}"))
    print(f"  ✓ checkpoint: {suffix}")


def _save_final(projector, llm, output_dir: str, clip_dim: int, llm_dim: int):
    torch.save(
        {
            "state_dict": unwrap(projector).state_dict(),
            "config": {"clip_dim": clip_dim, "llm_dim": llm_dim},
        },
        os.path.join(output_dir, "projector_final.pt"),
    )
    unwrap(llm).save_pretrained(os.path.join(output_dir, "lora_adapter_final"))
    print(f"  ✓ final artefacts written to {output_dir}/")


# ─── Per-worker training loop ─────────────────────────────────────────────────

def train_loop_per_worker(cfg: dict):
    """Executed inside each Ray worker (one per GPU slot)."""

    args = cfg["args"]  # plain dict mirroring argparse.Namespace

    rank       = ray.train.get_context().get_local_rank()
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    device     = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main    = world_rank == 0

    dtype      = resolve_dtype(args["dtype"])
    output_dir = args["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ── CLIP (always frozen) ──────────────────────────────────────────────────
    if is_main:
        print(f"Loading CLIP: {args['clip_model']}")
    clip_processor = CLIPProcessor.from_pretrained(args["clip_model"])
    clip_model = CLIPModel.from_pretrained(
        args["clip_model"], torch_dtype=dtype
    ).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    clip_dim = args["clip_dim"] or autodetect_clip_dim(clip_model)

    # ── LLM + LoRA ────────────────────────────────────────────────────────────
    if is_main:
        print(f"Loading LLM:  {args['llm_model']}")
    tokenizer = AutoTokenizer.from_pretrained(args["llm_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FSDP shards onto GPU after wrapping, so load to CPU first to avoid OOM.
    load_device = "cpu" if args["parallelism"] == "fsdp" else device
    llm_base = AutoModelForCausalLM.from_pretrained(
        args["llm_model"], torch_dtype=dtype
    ).to(load_device)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args["lora_r"],
        lora_alpha=args["lora_alpha"],
        lora_dropout=args["lora_dropout"],
        target_modules=args["lora_targets"],
        bias="none",
    )
    llm = get_peft_model(llm_base, lora_cfg)
    for name, p in llm.named_parameters():
        p.requires_grad = "lora_" in name
    llm.train()

    llm_dim = args["llm_dim"] or autodetect_llm_dim(llm)

    if is_main:
        lora_params = sum(p.numel() for p in llm.parameters() if p.requires_grad)
        print(f"  LoRA trainable params : {lora_params:,}")

    # ── Vision projector ──────────────────────────────────────────────────────
    projector = VisionProjection(clip_dim, llm_dim).to(device, dtype=dtype)
    projector.train()

    if is_main:
        proj_params = sum(p.numel() for p in projector.parameters())
        print(f"  Projector params      : {proj_params:,}")
        print(f"  Dims: CLIP={clip_dim}  LLM={llm_dim}")
        print(f"  Parallelism           : {args['parallelism'].upper()}")

    # ── Wrap for distributed training ─────────────────────────────────────────
    if args["parallelism"] == "fsdp":
        import functools
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # Identify the per-layer class for auto-wrapping (LLaMA / Mistral / Qwen …)
        try:
            decoder_cls = type(
                list(llm.base_model.model.model.layers)[0]
            )
            wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={decoder_cls},
            )
        except (AttributeError, IndexError):
            wrap_policy = None  # fall back to default FSDP wrapping

        llm_parallel = FSDP(
            llm.to(device),
            auto_wrap_policy=wrap_policy,
            device_id=rank,
        )
        # Projector is tiny – DDP is sufficient even in the FSDP run
        projector_parallel = DDP(projector, device_ids=[rank])

    else:  # ddp
        llm_parallel       = DDP(llm,       device_ids=[rank], find_unused_parameters=True)
        projector_parallel = DDP(projector, device_ids=[rank])

    # ── Dataset ───────────────────────────────────────────────────────────────
    if is_main:
        print(f"Loading dataset: {args['dataset']} / split={args['dataset_split']}")

    load_kwargs = {"split": args["dataset_split"]}
    if args["dataset_config"]:
        load_kwargs["name"] = args["dataset_config"]

    hf_data = load_dataset(args["dataset"], **load_kwargs)

    dataset = VisionTextDataset(
        hf_dataset=hf_data,
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        image_col=args["image_col"],
        text_col=args["text_col"],
        max_len=args["max_text_len"],
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=world_rank, shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        sampler=sampler,
        num_workers=args["cpus_per_worker"],
        pin_memory=True,
    )

    # ── Optimiser & LR schedule ───────────────────────────────────────────────
    trainable_params = (
        list(projector_parallel.parameters()) +
        [p for p in llm_parallel.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args["lr"], weight_decay=args["weight_decay"]
    )
    total_steps  = (len(loader) // args["grad_accum"]) * args["epochs"]
    warmup_steps = int(total_steps * args["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(args["epochs"]):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            loader, disable=not is_main,
            desc=f"Epoch {epoch + 1}/{args['epochs']}",
        )

        for step, batch in enumerate(pbar):

            with torch.autocast("cuda", dtype=dtype):
                loss = compute_loss(
                    batch, clip_model,
                    projector_parallel, llm_parallel,
                    tokenizer, device, dtype,
                )
                loss = loss / args["grad_accum"]

            loss.backward()
            epoch_loss += loss.item() * args["grad_accum"]

            if (step + 1) % args["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main:
                    pbar.set_postfix(
                        loss=f"{epoch_loss / (step + 1):.4f}",
                        step=global_step,
                    )

                if global_step % args["save_every"] == 0 and is_main:
                    _save_checkpoint(
                        projector_parallel, llm_parallel,
                        output_dir, suffix=f"step{global_step}",
                    )

        if is_main:
            _save_checkpoint(
                projector_parallel, llm_parallel,
                output_dir, suffix=f"epoch{epoch + 1}",
            )

        ray.train.report(
            metrics={"loss": epoch_loss / len(loader), "epoch": epoch + 1}
        )

    if is_main:
        _save_final(projector_parallel, llm_parallel, output_dir, clip_dim, llm_dim)
        print("Training complete.")


# ─── Inference helper ─────────────────────────────────────────────────────────

def load_for_inference(
    projector_path: str,
    lora_path: str,
    llm_model_id: str,
    dtype: str = "bf16",
    device: str = "cuda",
):
    """Reload the trained projector + LoRA-augmented LLM for inference."""
    pt_dtype  = resolve_dtype(dtype)
    ckpt      = torch.load(projector_path, map_location=device)
    cfg       = ckpt["config"]

    projector = VisionProjection(cfg["clip_dim"], cfg["llm_dim"])
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval().to(device, dtype=pt_dtype)

    base = AutoModelForCausalLM.from_pretrained(llm_model_id, torch_dtype=pt_dtype)
    llm  = PeftModel.from_pretrained(base, lora_path)
    llm.eval().to(device)

    return projector, llm


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"args": vars(args)},
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=True,
            resources_per_worker={
                "GPU": args.gpus_per_worker,
                "CPU": args.cpus_per_worker,
            },
        ),
        run_config=RunConfig(
            name=args.run_name,
            storage_path=str(Path(args.output_dir) / "ray_results"),
            checkpoint_config=CheckpointConfig(num_to_keep=args.checkpoints_to_keep),
        ),
    )

    result = trainer.fit()
    print("Ray training finished. Final metrics:", result.metrics)


if __name__ == "__main__":
    main()