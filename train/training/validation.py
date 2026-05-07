"""
Validation loop – runs one full pass over the validation DataLoader and
returns the mean loss.  Called at the end of every --val_every epoch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from training.loss import compute_loss


@torch.no_grad()
def run_validation(
    val_loader,
    clip_model: nn.Module,
    projector: nn.Module,
    llm: nn.Module,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    val_steps: int | None = None,
    is_main: bool = True,
) -> float:
    """
    Evaluate the model on the validation set.

    Parameters
    ----------
    val_loader  : DataLoader for the validation split.
    clip_model  : Frozen CLIP encoder.
    projector   : VisionProjection MLP.
    llm         : LoRA-augmented causal LM.
    tokenizer   : Tokeniser instance.
    device      : Target device.
    dtype       : Autocast dtype.
    val_steps   : Optional cap on the number of batches to evaluate.
                  None means evaluate the full loader.
    is_main     : Whether this is the rank-0 / primary worker (controls logging).

    Returns
    -------
    Mean validation loss (float) averaged over all evaluated batches.
    """
    projector.eval()
    llm.eval()

    total_loss   = 0.0
    total_batches = 0

    pbar = tqdm(val_loader, disable=not is_main, desc="  Validation")

    for step, batch in enumerate(pbar):
        if val_steps is not None and step >= val_steps:
            break

        with torch.autocast("cuda", dtype=dtype):
            loss = compute_loss(
                batch, clip_model,
                projector, llm,
                tokenizer, device, dtype,
            )

        total_loss    += loss.item()
        total_batches += 1
        pbar.set_postfix(val_loss=f"{total_loss / total_batches:.4f}")

    projector.train()
    llm.train()

    return total_loss / max(total_batches, 1)
