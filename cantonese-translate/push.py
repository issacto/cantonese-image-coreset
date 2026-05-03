# push_coreset_to_hf.py

import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi


def _flush_shard(images, captions, shard_idx, output_repo, hf_token):
    shard = Dataset.from_dict({
        "image": images,
        "caption": captions,
    })
    shard.push_to_hub(
        output_repo,
        token=hf_token,
        split="train",
        max_shard_size="500MB",
    )
    print(f"  [Shard {shard_idx}] pushed {len(images)} examples")


def push_coreset(
    hf_dataset: str,
    hf_split: str,
    hf_image_col: str,
    hf_caption_col: str,
    coreset_dir: str,
    output_repo: str,
    hf_token: str,
    push_batch_size: int = 200,
    hf_name: str = None,
):
    indices_path = Path(coreset_dir) / "coreset_indices.npy"
    coreset_ids = set(np.load(indices_path).tolist())
    print(f"[Info] Coreset size: {len(coreset_ids):,} indices")

    load_kwargs = dict(split=hf_split, streaming=True, token=hf_token)
    if hf_name:
        load_kwargs["name"] = hf_name

    ds = load_dataset(hf_dataset, **load_kwargs)

    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=output_repo,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    shard_idx = 0
    batch_images, batch_captions = [], []
    collected = 0

    for global_idx, example in enumerate(ds):
        if global_idx not in coreset_ids:
            continue

        img = example[hf_image_col]
        cap = example[hf_caption_col]

        if isinstance(cap, list):
            cap = cap[0] if cap else ""
        if isinstance(img, list):
            img = img[0] if img else None
        if img is None:
            continue

        batch_images.append(img)
        batch_captions.append(cap)
        collected += 1

        if len(batch_images) >= push_batch_size:
            _flush_shard(batch_images, batch_captions, shard_idx, output_repo, hf_token)
            shard_idx += 1
            batch_images.clear()
            batch_captions.clear()
            print(f"  [Progress] {collected:,} / {len(coreset_ids):,}")

        if collected == len(coreset_ids):
            print(f"[Info] All samples found at stream index {global_idx:,}")
            break

    if batch_images:
        _flush_shard(batch_images, batch_captions, shard_idx, output_repo, hf_token)
        print(f"  [Progress] {collected:,} / {len(coreset_ids):,}")

    print(f"\n[Done] Pushed {collected:,} pairs across {shard_idx + 1} shards")
    print(f"  https://huggingface.co/datasets/{output_repo}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dataset",      required=True)
    p.add_argument("--hf-split",        default="train")
    p.add_argument("--hf-image-col",    default="images")
    p.add_argument("--hf-caption-col",  default="texts")
    p.add_argument("--coreset-dir",     required=True)
    p.add_argument("--output-repo",     required=True)
    p.add_argument("--hf-token",        required=True)
    p.add_argument("--push-batch-size", type=int, default=200)
    p.add_argument("--hf-name",         default=None)
    args = p.parse_args()

    push_coreset(
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_image_col=args.hf_image_col,
        hf_caption_col=args.hf_caption_col,
        coreset_dir=args.coreset_dir,
        output_repo=args.output_repo,
        hf_token=args.hf_token,
        push_batch_size=args.push_batch_size,
        hf_name=args.hf_name,
    )