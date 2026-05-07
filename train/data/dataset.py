"""
Dataset utilities for vision-LLM training.

Image handling
──────────────
  Every row's image column is normalised to a **list** of PIL images regardless
  of whether the dataset stores a single image or a list.  All images are
  encoded by CLIP and returned as a stacked tensor of shape (N, C, H, W).

  A custom ``collate_fn`` (returned by ``make_collate_fn``) pads each batch to
  the maximum N in that batch and records ``num_images`` so the loss function
  can mask out the padding positions.

Supports
────────
  • Map-style HuggingFace datasets  (default)
  • Streaming / IterableDataset     (--streaming flag) – suited for large
    datasets that don't fit in RAM/disk.  The iterable is sharded across
    workers via datasets.distributed.split_dataset_by_node.

Validation
──────────
  A separate validation dataset can be supplied via --val_dataset / --val_split.
  When omitted the same repo is re-loaded with --val_split.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


# ─── Map-style dataset ────────────────────────────────────────────────────────

class VisionTextDataset(Dataset):
    """
    General-purpose wrapper for any HuggingFace image-caption dataset.

    Images are always collected into a list; single-image columns are wrapped
    in ``[image]`` automatically.  All images in the list are returned as a
    stacked tensor (N, C, H, W) so the model can attend to every image.
    """

    def __init__(
        self,
        hf_dataset,
        clip_processor,
        tokenizer,
        image_col: str,
        text_col: str,
        max_len: int = 128,
        text_selector: Optional[Callable] = None,
    ):
        self.data          = hf_dataset
        self.clip_proc     = clip_processor
        self.tokenizer     = tokenizer
        self.image_col     = image_col
        self.text_col      = text_col
        self.max_len       = max_len
        self.text_selector = text_selector or (
            lambda v: v[0] if isinstance(v, (list, tuple)) else v
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return _process_item(
            self.data[idx],
            self.clip_proc,
            self.tokenizer,
            self.image_col,
            self.text_col,
            self.max_len,
            self.text_selector,
        )


# ─── Streaming / IterableDataset wrapper ─────────────────────────────────────

class StreamingVisionTextDataset(IterableDataset):
    """
    Streaming wrapper that lazily processes rows from a HuggingFace
    IterableDataset.  Sharding across distributed workers is handled via
    ``datasets.distributed.split_dataset_by_node``.
    """

    def __init__(
        self,
        hf_iterable,
        clip_processor,
        tokenizer,
        image_col: str,
        text_col: str,
        max_len: int = 128,
        text_selector: Optional[Callable] = None,
        world_size: int = 1,
        world_rank: int = 0,
        buffer_size: int = 1000,
    ):
        super().__init__()
        self.clip_proc     = clip_processor
        self.tokenizer     = tokenizer
        self.image_col     = image_col
        self.text_col      = text_col
        self.max_len       = max_len
        self.text_selector = text_selector or (
            lambda v: v[0] if isinstance(v, (list, tuple)) else v
        )

        # Shard the stream so each worker sees a non-overlapping subset
        self.hf_iterable = split_dataset_by_node(
            hf_iterable.shuffle(buffer_size=buffer_size, seed=42),
            rank=world_rank,
            world_size=world_size,
        )

    def __iter__(self):
        for item in self.hf_iterable:
            yield _process_item(
                item,
                self.clip_proc,
                self.tokenizer,
                self.image_col,
                self.text_col,
                self.max_len,
                self.text_selector,
            )


# ─── Shared item processor ────────────────────────────────────────────────────

def _process_item(
    item: dict,
    clip_proc,
    tokenizer,
    image_col: str,
    text_col: str,
    max_len: int,
    text_selector: Callable,
) -> dict:
    """
    Convert one raw dataset row into model-ready tensors.

    The image column is always normalised to a list so the model can receive
    any number of visual tokens:

      • Single image  →  wrapped as [image]       → pixel_values: (1, C, H, W)
      • List of images → used as-is               → pixel_values: (N, C, H, W)

    Returns
    -------
    dict with keys:
        pixel_values   : FloatTensor (N, C, H, W)
        num_images     : int – number of valid images (N)
        input_ids      : LongTensor  (T,)
        attention_mask : LongTensor  (T,)
    """

    # ── Images → always a list ────────────────────────────────────────────────
    raw = item[image_col]
    if not isinstance(raw, (list, tuple)):
        raw = [raw]                     # single image → list of 1

    pixel_values_list: List[torch.Tensor] = []
    for img in raw:
        if isinstance(img, str):
            from PIL import Image as PILImage
            img = PILImage.open(img)
        img = img.convert("RGB")
        pv  = clip_proc(images=img, return_tensors="pt").pixel_values.squeeze(0)
        pixel_values_list.append(pv)    # (C, H, W)

    pixel_values = torch.stack(pixel_values_list)   # (N, C, H, W)
    num_images   = len(pixel_values_list)

    # ── Text ─────────────────────────────────────────────────────────────────
    caption  = text_selector(item[text_col])
    encoding = tokenizer(
        caption,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    return {
        "pixel_values":   pixel_values,                         # (N, C, H, W)
        "num_images":     num_images,                           # int
        "input_ids":      encoding.input_ids.squeeze(0),        # (T,)
        "attention_mask": encoding.attention_mask.squeeze(0),   # (T,)
    }


# ─── Collate function ─────────────────────────────────────────────────────────

def make_collate_fn():
    """
    Return a collate function that pads ``pixel_values`` to the maximum number
    of images in each batch.

    Within a batch different rows may have different N (number of images).
    Padding positions are filled with zeros; ``num_images`` tells the loss
    function how many visual tokens are real so it can build the correct
    attention mask.

    Batch output keys
    -----------------
    pixel_values   : FloatTensor  (B, N_max, C, H, W)
    num_images     : LongTensor   (B,)
    input_ids      : LongTensor   (B, T)
    attention_mask : LongTensor   (B, T)
    """
    def collate_fn(batch):
        max_n  = max(item["num_images"] for item in batch)
        sample = batch[0]["pixel_values"]           # (n, C, H, W)
        _, C, H, W = sample.shape

        padded = torch.zeros(len(batch), max_n, C, H, W, dtype=sample.dtype)
        for i, item in enumerate(batch):
            n = item["num_images"]
            padded[i, :n] = item["pixel_values"]

        return {
            "pixel_values":   padded,                                           # (B, N_max, C, H, W)
            "num_images":     torch.tensor([item["num_images"] for item in batch], dtype=torch.long),
            "input_ids":      torch.stack([item["input_ids"]      for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        }
    return collate_fn


# ─── DataLoader builders ──────────────────────────────────────────────────────

def build_train_loader(
    args: dict,
    clip_processor,
    tokenizer,
    world_size: int,
    world_rank: int,
) -> DataLoader:
    """Return a DataLoader for the training split."""

    load_kwargs   = {"split": args["dataset_split"]}
    if args.get("dataset_config"):
        load_kwargs["name"] = args["dataset_config"]

    train_samples = args.get("train_samples")

    text_subfield = args.get("text_subfield")
    if text_subfield:
        text_selector = lambda v, _sf=text_subfield: (
            v[0][_sf] if isinstance(v, (list, tuple)) else v[_sf]
        )
    else:
        text_selector = None

    collate_fn = make_collate_fn()

    if args.get("streaming"):
        hf_data = load_dataset(args["dataset"], streaming=True, **load_kwargs)
        if train_samples is not None:
            hf_data = hf_data.take(train_samples)
        dataset = StreamingVisionTextDataset(
            hf_iterable=hf_data,
            clip_processor=clip_processor,
            tokenizer=tokenizer,
            image_col=args["image_col"],
            text_col=args["text_col"],
            max_len=args["max_text_len"],
            text_selector=text_selector,
            world_size=world_size,
            world_rank=world_rank,
            buffer_size=args.get("streaming_buffer_size", 1000),
        )
        return DataLoader(
            dataset,
            batch_size=args["batch_size"],
            num_workers=args["cpus_per_worker"],
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        hf_data = load_dataset(args["dataset"], **load_kwargs)
        if train_samples is not None:
            hf_data = hf_data.select(range(min(train_samples, len(hf_data))))
        dataset = VisionTextDataset(
            hf_dataset=hf_data,
            clip_processor=clip_processor,
            tokenizer=tokenizer,
            image_col=args["image_col"],
            text_col=args["text_col"],
            max_len=args["max_text_len"],
            text_selector=text_selector,
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=world_rank, shuffle=True,
        )
        return DataLoader(
            dataset,
            batch_size=args["batch_size"],
            sampler=sampler,
            num_workers=args["cpus_per_worker"],
            pin_memory=True,
            collate_fn=collate_fn,
        )


def build_val_loader(
    args: dict,
    clip_processor,
    tokenizer,
    world_size: int,
    world_rank: int,
) -> Optional[DataLoader]:
    """
    Return a DataLoader for the validation split, or None if no val set is
    configured.
    """
    val_split = args.get("val_split")
    val_repo  = args.get("val_dataset")

    if val_split is None and val_repo is None:
        return None

    val_split  = val_split or "validation"
    val_repo   = val_repo or args["dataset"]
    val_config = args.get("val_dataset_config") or args.get("dataset_config")

    load_kwargs = {"split": val_split}
    if val_config:
        load_kwargs["name"] = val_config

    val_samples   = args.get("val_samples")
    text_subfield = args.get("text_subfield")
    if text_subfield:
        text_selector = lambda v, _sf=text_subfield: (
            v[0][_sf] if isinstance(v, (list, tuple)) else v[_sf]
        )
    else:
        text_selector = None

    collate_fn = make_collate_fn()

    try:
        if args.get("streaming"):
            hf_val = load_dataset(val_repo, streaming=True, **load_kwargs)
            if val_samples is not None:
                hf_val = hf_val.take(val_samples)
            dataset = StreamingVisionTextDataset(
                hf_iterable=hf_val,
                clip_processor=clip_processor,
                tokenizer=tokenizer,
                image_col=args["image_col"],
                text_col=args["text_col"],
                max_len=args["max_text_len"],
                text_selector=text_selector,
                world_size=world_size,
                world_rank=world_rank,
                buffer_size=args.get("streaming_buffer_size", 1000),
            )
            return DataLoader(
                dataset,
                batch_size=args["batch_size"],
                num_workers=args["cpus_per_worker"],
                pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            hf_val = load_dataset(val_repo, **load_kwargs)
            if val_samples is not None:
                hf_val = hf_val.select(range(min(val_samples, len(hf_val))))
            dataset = VisionTextDataset(
                hf_dataset=hf_val,
                clip_processor=clip_processor,
                tokenizer=tokenizer,
                image_col=args["image_col"],
                text_col=args["text_col"],
                max_len=args["max_text_len"],
                text_selector=text_selector,
            )
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=world_rank, shuffle=False,
            )
            return DataLoader(
                dataset,
                batch_size=args["batch_size"],
                sampler=sampler,
                num_workers=args["cpus_per_worker"],
                pin_memory=True,
                collate_fn=collate_fn,
            )

    except Exception as exc:
        print(f"[WARNING] Could not load validation split '{val_split}' "
              f"from '{val_repo}': {exc}. Skipping validation.")
        return None