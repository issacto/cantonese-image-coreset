"""
Forward pass and loss computation for the vision-LLM model.

Change from previous version
─────────────────────────────
The batch now carries a ``labels`` tensor produced by the data pipeline.
Labels already have -100 at:
  • visual prefix positions  (added here, as before)
  • prompt / question tokens (masked in data.py so the model only learns
                              to predict the final assistant answer)
  • padding positions        (masked in data.py)

This means the model is supervised ONLY on the assistant answer tokens,
not on "User:" / "Assistant:" delimiters or prior conversation history.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
    Prepend projected visual patch tokens to the token embeddings and compute
    next-token prediction loss, supervised only on assistant answer tokens.

    Args
    ----
    batch       : Dict with keys:
                    pixel_values   (B, N, C, H, W)
                    num_images     (B,)
                    input_ids      (B, T)
                    attention_mask (B, T)
                    labels         (B, T)  ← pre-masked by data pipeline
    clip_model  : Frozen CLIP model.
    projector   : VisionProjection MLP (trainable).
    llm         : LoRA-augmented causal LM (trainable).
    tokenizer   : Tokenizer instance.
    device      : Target device.
    dtype       : Autocast dtype.

    Returns
    -------
    Scalar loss tensor.
    """
    pixel_values   = batch["pixel_values"].to(device, dtype=dtype)  # (B, N, C, H, W)
    num_images     = batch["num_images"].to(device)                  # (B,)
    input_ids      = batch["input_ids"].to(device)                   # (B, T)
    attention_mask = batch["attention_mask"].to(device)              # (B, T)
    text_labels    = batch["labels"].to(device)                      # (B, T)

    B, N, C, H, W = pixel_values.shape

    # ── CLIP vision encoder (frozen) ──────────────────────────────────────────
    pv_flat = pixel_values.view(B * N, C, H, W)
    with torch.no_grad():
        vision_out     = clip_model.vision_model(pixel_values=pv_flat)
        image_features = vision_out.last_hidden_state[:, 1:, :].to(dtype)
        # (B*N, P, clip_dim)  — CLS dropped

    P        = image_features.shape[1]
    clip_dim = image_features.shape[2]

    image_features = image_features.view(B, N, P, clip_dim)   # (B, N, P, clip_dim)

    # ── Project into LLM token space ──────────────────────────────────────────
    visual_tokens = projector(image_features)                  # (B, N, P, llm_dim)
    llm_dim       = visual_tokens.shape[-1]
    visual_tokens = visual_tokens.view(B, N * P, llm_dim)     # (B, N*P, llm_dim)

    # ── Text embeddings ───────────────────────────────────────────────────────
    inner = llm
    for attr in ("module", "base_model", "model"):
        inner = getattr(inner, attr, inner)

    with torch.no_grad():
        text_embeds = inner.get_input_embeddings()(input_ids)  # (B, T, llm_dim)

    # ── Concatenate visual prefix + text ──────────────────────────────────────
    inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)  # (B, N*P+T, llm_dim)

    # ── Attention mask ────────────────────────────────────────────────────────
    token_image_idx = torch.arange(N * P, device=device).unsqueeze(0) // P
    visual_mask     = (token_image_idx < num_images.unsqueeze(1)).to(attention_mask.dtype)
    full_mask       = torch.cat([visual_mask, attention_mask], dim=1)   # (B, N*P+T)

    # ── Labels ────────────────────────────────────────────────────────────────
    # Visual prefix is never supervised (-100).
    # text_labels already has -100 for prompt tokens and padding (from data.py).
    visual_labels = torch.full((B, N * P), -100, dtype=torch.long, device=device)
    labels        = torch.cat([visual_labels, text_labels], dim=1)      # (B, N*P+T)

    output = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels,
    )
    return output.loss