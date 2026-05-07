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

Dataset modes
─────────────
  default   – loads the full dataset into memory / disk cache (HuggingFace
              Arrow format). Fast iteration but requires sufficient disk space.
  streaming – pass --streaming to lazily stream batches from the Hub without
              downloading the full dataset. Recommended for large datasets.

Validation
──────────
  Pass --val_split (e.g. "validation", "test") to evaluate on a split of the
  same dataset, or --val_dataset to use a completely separate HuggingFace repo.
  Validation runs every --val_every epochs (default: 1).

Usage
─────
  # Minimal – single node, all local GPUs, DDP:
  python train_ray_lora.py \\
      --clip_model openai/clip-vit-base-patch32 \\
      --llm_model  hon9kon9ize/CantoneseLLMChat-v1.0-7B \\
      --dataset    Issactoto/flickr8k-cantonese \\
      --image_col  image \\
      --text_col   yue_caption \\
      --val_split  validation

  # Streaming + separate val dataset, multi-node FSDP:
  RAY_ADDRESS=ray://<head>:10001 python train_ray_lora.py \\
      --clip_model       openai/clip-vit-large-patch14 \\
      --llm_model        meta-llama/Llama-3-8b-hf \\
      --dataset          nlphuji/flickr30k \\
      --image_col        image \\
      --text_col         caption \\
      --streaming \\
      --val_dataset      nlphuji/flickr30k \\
      --val_split        test \\
      --val_every        1 \\
      --val_steps        200 \\
      --parallelism      fsdp \\
      --num_workers      8 \\
      --gpus_per_worker  1 \\
      --cpus_per_worker  4 \\
      --batch_size       2 \\
      --lora_r           32 \\
      --epochs           5 \\
      --output_dir       ./my_checkpoints

Dependencies
────────────
  pip install "ray[train]>=2.10" torch transformers datasets peft tqdm pillow
"""

from pathlib import Path

import ray
from ray.train import ScalingConfig, CheckpointConfig, RunConfig
from ray.train.torch import TorchTrainer

from core import parse_args
from training import train_loop_per_worker


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
            storage_path=str(Path(args.output_dir).resolve() / "ray_results"),
            # checkpoint_config=CheckpointConfig(num_to_keep=args.checkpoints_to_keep),
        ),
    )

    result = trainer.fit()
    print("Ray training finished. Final metrics:", result.metrics)


if __name__ == "__main__":
    main()
