"""
Checkpoint save and load helpers.
"""

import os

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from core.models import VisionProjection
from core.utils import unwrap_ddp_fsdp, resolve_dtype


# ─── Saving ───────────────────────────────────────────────────────────────────

def save_step_checkpoint(
    projector,
    llm,
    output_dir: str,
    suffix: str,
) -> None:
    """Save projector weights and LoRA adapter for a mid-training checkpoint."""
    torch.save(
        unwrap_ddp_fsdp(projector).state_dict(),
        os.path.join(output_dir, f"projector_{suffix}.pt"),
    )
    unwrap_ddp_fsdp(llm).save_pretrained(
        os.path.join(output_dir, f"lora_adapter_{suffix}")
    )
    print(f"  ✓ checkpoint saved: {suffix}")


def save_final(
    projector,
    llm,
    output_dir: str,
    clip_dim: int,
    llm_dim: int,
) -> None:
    """
    Save the final projector (with config embedded) and LoRA adapter.
    The config dict lets ``load_for_inference`` reconstruct the MLP without
    needing the original CLI args.
    """
    torch.save(
        {
            "state_dict": unwrap_ddp_fsdp(projector).state_dict(),
            "config": {"clip_dim": clip_dim, "llm_dim": llm_dim},
        },
        os.path.join(output_dir, "projector_final.pt"),
    )
    unwrap_ddp_fsdp(llm).save_pretrained(
        os.path.join(output_dir, "lora_adapter_final")
    )
    print(f"  ✓ final artefacts written to {output_dir}/")


# ─── Loading (inference) ──────────────────────────────────────────────────────

def load_for_inference(
    projector_path: str,
    lora_path: str,
    llm_model_id: str,
    dtype: str = "bf16",
    device: str = "cuda",
):
    """
    Reload the trained projector + LoRA-augmented LLM for inference.

    Parameters
    ----------
    projector_path : Path to ``projector_final.pt`` (or any step checkpoint).
    lora_path      : Directory containing the saved LoRA adapter.
    llm_model_id   : HuggingFace model ID or local path for the base LLM.
    dtype          : One of "bf16", "fp16", "fp32".
    device         : Target device string.

    Returns
    -------
    (projector, llm) – both in eval mode, moved to ``device``.
    """
    pt_dtype = resolve_dtype(dtype)
    ckpt     = torch.load(projector_path, map_location=device)
    cfg      = ckpt["config"]

    projector = VisionProjection(cfg["clip_dim"], cfg["llm_dim"])
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval().to(device, dtype=pt_dtype)

    base = AutoModelForCausalLM.from_pretrained(llm_model_id, torch_dtype=pt_dtype)
    llm  = PeftModel.from_pretrained(base, lora_path)
    llm.eval().to(device)

    return projector, llm
