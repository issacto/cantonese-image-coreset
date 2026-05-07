"""
Forward pass and loss computation for the vision-LLM model.

Multi-image support
───────────────────
``pixel_values`` is expected to be shape **(B, N, C, H, W)** where N is the
(padded) number of images per sample.  ``num_images`` is a LongTensor of shape
(B,) that records how many of those N slots are real; the rest are zero-padded
and masked out of the attention mask.

For datasets with a single image per row the DataLoader will produce
``pixel_values`` of shape (B, 1, C, H, W) and ``num_images`` will be all-ones,
so the behaviour is identical to the previous single-image code path.
"""

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
    Prepend projected visual tokens (one per image) to caption token embeddings,
    then compute next-token prediction loss.

    All visual-token positions are masked with -100 in ``labels`` so the model
    is only supervised on the caption tokens.

    Args
    ----
    batch       : Dict with keys:
                    pixel_values   (B, N, C, H, W)
                    num_images     (B,)  – number of real images per sample
                    input_ids      (B, T)
                    attention_mask (B, T)
    clip_model  : Frozen CLIP model (vision encoder only used).
    projector   : VisionProjection MLP (trainable).
    llm         : LoRA-augmented causal LM (LoRA weights trainable).
    tokenizer   : Tokenizer – used to mask pad tokens in labels.
    device      : Target CUDA/CPU device.
    dtype       : Autocast dtype (bf16 / fp16 / fp32).

    Returns
    -------
    Scalar loss tensor.
    """
    pixel_values   = batch["pixel_values"].to(device, dtype=dtype)  # (B, N, C, H, W)
    num_images     = batch["num_images"].to(device)                  # (B,)
    input_ids      = batch["input_ids"].to(device)                   # (B, T)
    attention_mask = batch["attention_mask"].to(device)              # (B, T)

    B, N, C, H, W = pixel_values.shape

    # ── CLIP vision encoder (frozen) – encode all N images at once ────────────
    # Flatten to (B*N, C, H, W) for a single batched forward pass
    pv_flat = pixel_values.view(B * N, C, H, W)
    with torch.no_grad():
        image_features = clip_model.vision_model(
            pixel_values=pv_flat
        ).pooler_output.to(dtype)                               # (B*N, clip_dim)

    image_features = image_features.view(B, N, -1)             # (B, N, clip_dim)

    # ── Project all image features into LLM token space ──────────────────────
    visual_tokens = projector(image_features)                   # (B, N, llm_dim)

    # ── Reach embed_tokens through DDP / FSDP / PEFT wrapping ────────────────
    inner = llm
    for attr in ("module", "base_model", "model"):
        inner = getattr(inner, attr, inner)

    with torch.no_grad():
        text_embeds = inner.get_input_embeddings()(input_ids)

    # ── Concatenate visual prefix + text embeddings ───────────────────────────
    inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)  # (B, N+T, D)

    # Build visual attention mask: 1 for real image slots, 0 for padding
    # num_images[i] tells us how many of the N slots are valid for sample i
    visual_mask = torch.arange(N, device=device).unsqueeze(0) < num_images.unsqueeze(1)
    visual_mask = visual_mask.to(attention_mask.dtype)          # (B, N)

    full_mask = torch.cat([visual_mask, attention_mask], dim=1) # (B, N+T)

    # Mask all visual prefix positions in labels (not supervised)
    visual_labels = torch.full((B, N), -100, dtype=torch.long, device=device)
    labels        = torch.cat([visual_labels, input_ids], dim=1)    # (B, N+T)
    labels[labels == tokenizer.pad_token_id] = -100

    output = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels,
    )
    return output.loss