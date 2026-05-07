"""
Model definitions for the vision-LLM bridge.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel


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


# ─── Dim auto-detection ───────────────────────────────────────────────────────

def autodetect_clip_dim(clip_model: CLIPModel) -> int:
    return clip_model.config.vision_config.hidden_size


def autodetect_llm_dim(llm: nn.Module) -> int:
    from core.utils import unwrap
    cfg = unwrap(llm).config
    return getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
