"""
CLI argument definitions for vision-LLM training.
"""

import argparse


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
    d.add_argument("--val_dataset",        default=None,
                   help="HuggingFace dataset repo ID or local path for validation. "
                        "Falls back to --dataset if omitted.")
    d.add_argument("--val_split",          default=None,
                   help="Validation split name (e.g. 'validation', 'test', 'train[80%%:]'). "
                        "Omit entirely to skip validation.")
    d.add_argument("--val_dataset_config", default=None,
                   help="Optional dataset config/subset for the validation set. "
                        "Defaults to --dataset_config.")
    d.add_argument("--val_every",          type=int, default=1,
                   help="Run validation every N epochs.")
    d.add_argument("--val_steps",          type=int, default=None,
                   help="Cap validation at this many batches. None = full val set.")
    d.add_argument("--image_col",      required=True,
                   help="Column that contains PIL images or file-path strings. "
                        "Single images and lists of images are both supported — "
                        "all images in the column are used as visual tokens.")
    d.add_argument("--text_col",       required=True,
                   help="Column that contains caption / text target strings. "
                        "If a row holds a list of captions the first is used.")
    d.add_argument("--text_subfield",  default=None,
                   help="When --text_col contains a list of dicts (e.g. Docmatix / FineVisionMax "
                        "'texts' column), name the dict key to extract as the text target. "
                        "E.g. --text_subfield assistant  ->  row['texts'][0]['assistant'].")
    d.add_argument("--max_text_len",   type=int, default=128,
                   help="Maximum tokenised text length (tokens).")
    d.add_argument("--train_samples",  type=int, default=None,
                   help="Cap the training set at this many examples. "
                        "None = use the full split.")
    d.add_argument("--val_samples",    type=int, default=None,
                   help="Cap the validation set at this many examples.")
    d.add_argument("--streaming",      action="store_true",
                   help="Enable HuggingFace dataset streaming (iterable). "
                        "Recommended for large datasets that don't fit in RAM/disk.")
    d.add_argument("--streaming_buffer_size", type=int, default=1000,
                   help="Shuffle buffer size when streaming is enabled.")

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