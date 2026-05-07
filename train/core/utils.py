"""
Shared utility helpers used across training modules.
"""

import torch
import torch.nn as nn


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def unwrap(model: nn.Module) -> nn.Module:
    """Strip one layer of DDP / FSDP / PEFT wrapping to reach the raw module."""
    inner = model
    for attr in ("module", "base_model", "model"):
        inner = getattr(inner, attr, inner)
    return inner


def unwrap_ddp_fsdp(model: nn.Module) -> nn.Module:
    """Strip only the outermost DDP / FSDP wrapper (one level)."""
    return model.module if hasattr(model, "module") else model
