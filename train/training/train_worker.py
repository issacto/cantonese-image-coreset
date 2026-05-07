"""
Per-worker training loop executed inside each Ray worker (one per GPU slot).

Responsibilities
────────────────
  • Load CLIP, LLM, tokeniser.
  • Build LoRA-augmented LLM + VisionProjection.
  • Wrap models for DDP or FSDP distributed training.
  • Build train + (optional) validation DataLoaders.
  • Run the training loop with gradient accumulation, LR scheduling,
    periodic checkpointing, and per-epoch validation.
  • Report metrics to Ray Train after every epoch.
"""

from __future__ import annotations

import os
import tempfile  # ← new

import ray
import ray.train
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPModel,
    CLIPProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from ray.train import Checkpoint  # ← new

from core import (
    VisionProjection,
    autodetect_clip_dim,
    autodetect_llm_dim,
    resolve_dtype,
)
from data import build_train_loader, build_val_loader
from training.loss import compute_loss
from training.checkpoint import save_step_checkpoint, save_final
from training.validation import run_validation


def train_loop_per_worker(cfg: dict):
    """Executed inside each Ray worker (one per GPU slot)."""

    args = cfg["args"]   # plain dict mirroring argparse.Namespace

    rank       = ray.train.get_context().get_local_rank()
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    # FSDP shards onto GPU after wrapping → load to CPU first to avoid OOM.
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
        if args.get("streaming"):
            print("  Dataset mode          : STREAMING")

    # ── Wrap for distributed training ─────────────────────────────────────────
    if args["parallelism"] == "fsdp":
        import functools
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        try:
            decoder_cls = type(list(llm.base_model.model.model.layers)[0])
            wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={decoder_cls},
            )
        except (AttributeError, IndexError):
            wrap_policy = None  # fall back to default FSDP wrapping

        llm_parallel       = FSDP(llm.to(device), auto_wrap_policy=wrap_policy, device_id=rank)
        projector_parallel = DDP(projector, device_ids=[rank])
    else:
        llm_parallel       = DDP(llm,       device_ids=[rank], find_unused_parameters=True)
        projector_parallel = DDP(projector, device_ids=[rank])

    # ── DataLoaders ───────────────────────────────────────────────────────────
    if is_main:
        print(f"Loading training dataset  : {args['dataset']} / {args['dataset_split']}")
    train_loader = build_train_loader(args, clip_processor, tokenizer, world_size, world_rank)

    val_loader = None
    if args.get("val_split") or args.get("val_dataset"):
        val_src   = args.get("val_dataset") or args["dataset"]
        val_split = args.get("val_split") or "validation"
        if is_main:
            print(f"Loading validation dataset: {val_src} / {val_split}")
        val_loader = build_val_loader(args, clip_processor, tokenizer, world_size, world_rank)

    # ── Optimiser & LR schedule ───────────────────────────────────────────────
    trainable_params = (
        list(projector_parallel.parameters()) +
        [p for p in llm_parallel.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args["lr"], weight_decay=args["weight_decay"]
    )

    # For streaming datasets we don't know len(loader) upfront; fall back to
    # a rough estimate (can be overridden via --save_every which controls ckpts).
    try:
        steps_per_epoch = len(train_loader) // args["grad_accum"]
    except TypeError:
        # IterableDataset has no __len__
        steps_per_epoch = args.get("streaming_steps_per_epoch", 1000)
        if is_main:
            print(f"  [streaming] Assuming ~{steps_per_epoch} optimiser steps/epoch "
                  f"for LR schedule. Override with --streaming_steps_per_epoch.")

    total_steps  = steps_per_epoch * args["epochs"]
    warmup_steps = int(total_steps * args["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    val_every   = args.get("val_every", 1)
    val_steps   = args.get("val_steps")   # None = full val set

    for epoch in range(args["epochs"]):

        # DistributedSampler needs epoch for seeding; streaming handles its own
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss    = 0.0
        epoch_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader, disable=not is_main,
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
            epoch_loss    += loss.item() * args["grad_accum"]
            epoch_batches += 1

            if (step + 1) % args["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main:
                    pbar.set_postfix(
                        loss=f"{epoch_loss / epoch_batches:.4f}",
                        step=global_step,
                    )

                if global_step % args["save_every"] == 0 and is_main:
                    save_step_checkpoint(
                        projector_parallel, llm_parallel,
                        output_dir, suffix=f"step{global_step}",
                    )

        # ── End-of-epoch checkpoint ───────────────────────────────────────────
        ckpt = None
        if is_main:
            epoch_dir = os.path.join(output_dir, f"epoch{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)
            save_step_checkpoint(
                projector_parallel, llm_parallel,
                epoch_dir, suffix=f"epoch{epoch + 1}",
            )
            ckpt = Checkpoint.from_directory(epoch_dir)


        train_loss = epoch_loss / max(epoch_batches, 1)
        metrics    = {"loss": train_loss, "epoch": epoch + 1}

        # ── Validation ────────────────────────────────────────────────────────
        if val_loader is not None and (epoch + 1) % val_every == 0:
            val_loss = run_validation(
                val_loader=val_loader,
                clip_model=clip_model,
                projector=projector_parallel,
                llm=llm_parallel,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                val_steps=val_steps,
                is_main=is_main,
            )
            metrics["val_loss"] = val_loss
            if is_main:
                print(f"  [Epoch {epoch + 1}] train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}")

        ray.train.report(metrics=metrics, checkpoint=ckpt)

    # ── Final save (rank 0 only) ──────────────────────────────────────────────
    if is_main:
        save_final(projector_parallel, llm_parallel, output_dir, clip_dim, llm_dim)
        print("Training complete.")