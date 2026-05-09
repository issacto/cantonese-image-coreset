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
  • Streaming / IterableDataset     (--streaming flag)

Validation
──────────
  A separate validation dataset can be supplied via --val_dataset / --val_split.
  When omitted the same repo is re-loaded with --val_split.

Causal conversation formatting  (when --text_subfield is set)
─────────────────────────────────────────────────────────────
  Given N QA pairs per document, we produce N training examples per row,
  each one adding one more turn of context:

    Example 1 : User: Q1\\nAssistant: A1
    Example 2 : User: Q1\\nAssistant: A1\\nUser: Q2\\nAssistant: A2
    Example 3 : User: Q1\\nAssistant: A1\\n...\\nUser: Q3\\nAssistant: A3
    ...

  Labels mask everything except the FINAL assistant answer with -100, so the
  model only learns to predict new information given the image + prior context.

  The dataset therefore has len(rows) * avg_pairs_per_row effective examples.
  VisionTextDataset flattens these into a single indexed list at init time.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


# ─── Config auto-detection ────────────────────────────────────────────────────

def _resolve_config(repo: str, preferred: Optional[str]) -> Optional[str]:
    if preferred is not None:
        return preferred
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(repo)
        return configs[0] if configs else None
    except Exception:
        return None


# ─── Conversation builder ─────────────────────────────────────────────────────

def _build_conversation_examples(
    pairs: list,
    subfield: str,
) -> List[Tuple[str, int]]:
    """
    Given a list of {user, <subfield>} dicts, produce one (text, answer_start)
    tuple per turn using causal accumulation.

    Returns
    -------
    List of (full_text, answer_char_start) tuples where answer_char_start is
    the character index in full_text where the FINAL assistant answer begins.
    Only tokens from answer_char_start onward are supervised.

    Example with 3 pairs
    --------------------
    Turn 1:
        text  = "User: Q1\\nAssistant: A1"
        start = len("User: Q1\\nAssistant: ")

    Turn 2:
        text  = "User: Q1\\nAssistant: A1\\nUser: Q2\\nAssistant: A2"
        start = len("User: Q1\\nAssistant: A1\\nUser: Q2\\nAssistant: ")

    Turn 3:
        text  = "User: Q1\\nAssistant: A1\\n...\\nUser: Q3\\nAssistant: A3"
        start = len("...\\nUser: Q3\\nAssistant: ")
    """
    valid   = []
    invalid = []
    for p in pairs:
        missing = [k for k in ("user", subfield) if not p.get(k)]
        if missing:
            invalid.append((p, missing))
        else:
            valid.append(p)

    for bad_pair, missing_keys in invalid:
        warnings.warn(
            f"Skipping QA pair — missing keys {missing_keys}. "
            f"Pair preview: { {k: str(v)[:60] for k, v in bad_pair.items()} }"
        )

    if not valid:
        warnings.warn(
            f"Row has no valid QA pairs for subfield='{subfield}' — "
            f"skipping entire row. Total pairs in row: {len(pairs)}."
        )
        return []

    examples = []
    history  = ""   # accumulates prior turns

    for pair in valid:
        prompt_part = f"{history}User: {pair['user']}\nAssistant: "
        answer_part = pair[subfield]
        full_text   = prompt_part + answer_part

        examples.append((full_text, len(prompt_part)))
        # Add this turn to history for the next example
        history = full_text + "\n"

    return examples


# ─── Shared item processor ────────────────────────────────────────────────────

def _process_images(item: dict, image_col: str, clip_proc) -> Optional[torch.Tensor]:
    """
    Load and CLIP-process all images for a row.
    Returns pixel_values (N, C, H, W) or None if all images failed.
    """
    raw = item[image_col]
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    raw = [img for img in raw if img is not None]

    if not raw:
        return None

    pixel_values_list = []
    for img in raw:
        try:
            if isinstance(img, str):
                from PIL import Image as PILImage
                img = PILImage.open(img)
            img = img.convert("RGB")
            pv  = clip_proc(images=img, return_tensors="pt").pixel_values.squeeze(0)
            pixel_values_list.append(pv)
        except Exception as exc:
            warnings.warn(f"Skipping one image due to error: {exc}")

    if not pixel_values_list:
        return None

    return torch.stack(pixel_values_list)   # (N, C, H, W)


def _tokenize_with_label_mask(
    full_text: str,
    answer_char_start: int,
    tokenizer,
    max_len: int,
) -> Optional[dict]:
    eos = tokenizer.eos_token or ""
    full_text_with_eos = full_text + eos
    prompt_text = full_text[:answer_char_start]

    # Step 1 — tokenize full sequence (truncated) to get the actual input_ids
    encoding = tokenizer(
        full_text_with_eos,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)

    # Step 2 — tokenize prompt alone (truncated to same max_len) so that
    # prompt_token_len is measured on the same token sequence as input_ids.
    # Previously this used truncation=False, so on long causal-history turns
    # prompt_token_len > len(input_ids), masking the entire sequence → dropped.
    prompt_ids = tokenizer(
        prompt_text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=False,
    )["input_ids"]
    prompt_token_len = len(prompt_ids)

    # Step 3 — build labels: mask prompt and padding, supervise answer tokens
    labels = input_ids.clone()
    labels[:prompt_token_len] = -100
    labels[input_ids == tokenizer.pad_token_id] = -100

    if (labels != -100).sum() == 0:
        return None

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }
# ─── Map-style dataset ────────────────────────────────────────────────────────

class VisionTextDataset(Dataset):
    """
    Flattens all causal conversation examples across all rows into a single
    indexed list at init time.

    Each entry in self.examples is:
        (row_idx, full_text, answer_char_start)

    For plain-text datasets (no text_subfield) each row produces exactly one
    example with answer_char_start=0 (full text supervised).
    """

    def __init__(
        self,
        hf_dataset,
        clip_processor,
        tokenizer,
        image_col: str,
        text_col: str,
        max_len: int = 512,
        text_subfield: Optional[str] = None,
    ):
        self.data          = hf_dataset
        self.clip_proc     = clip_processor
        self.tokenizer     = tokenizer
        self.image_col     = image_col
        self.text_col      = text_col
        self.max_len       = max_len
        self.text_subfield = text_subfield

        # Pre-build the flat example index
        # (row_idx, full_text, answer_char_start)
        print("  Indexing conversation examples …")
        self.examples: List[Tuple[int, str, int]] = []

        for row_idx in range(len(hf_dataset)):
            raw_text = hf_dataset[row_idx][text_col]

            if text_subfield:
                pairs = raw_text if isinstance(raw_text, (list, tuple)) else [raw_text]
                convs = _build_conversation_examples(pairs, text_subfield)
                for full_text, answer_start in convs:
                    self.examples.append((row_idx, full_text, answer_start))
            else:
                # Plain caption — full text is supervised
                caption = raw_text[0] if isinstance(raw_text, (list, tuple)) else raw_text
                self.examples.append((row_idx, caption, 0))

        print(f"  Indexed {len(self.examples):,} examples from {len(hf_dataset):,} rows.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        start_idx = idx
        while True:
            row_idx, full_text, answer_char_start = self.examples[idx]

            pixel_values = _process_images(self.data[row_idx], self.image_col, self.clip_proc)
            if pixel_values is None:
                idx = (idx + 1) % len(self.examples)
                if idx == start_idx:
                    raise RuntimeError("No valid items found in dataset.")
                continue

            tok = _tokenize_with_label_mask(
                full_text, answer_char_start, self.tokenizer, self.max_len
            )
            if tok is None:
                idx = (idx + 1) % len(self.examples)
                if idx == start_idx:
                    raise RuntimeError("No valid items found in dataset.")
                continue

            return {
                "pixel_values":   pixel_values,
                "num_images":     pixel_values.shape[0],
                **tok,
            }


# ─── Streaming / IterableDataset wrapper ─────────────────────────────────────

class StreamingVisionTextDataset(IterableDataset):
    """
    Streaming wrapper. Each row yields N examples (one per QA turn).
    """

    def __init__(
        self,
        hf_iterable,
        clip_processor,
        tokenizer,
        image_col: str,
        text_col: str,
        max_len: int = 512,
        text_subfield: Optional[str] = None,
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
        self.text_subfield = text_subfield

        self.hf_iterable = split_dataset_by_node(
            hf_iterable.shuffle(buffer_size=buffer_size, seed=42),
            rank=world_rank,
            world_size=world_size,
        )

    def __iter__(self):
        for item in self.hf_iterable:
            pixel_values = _process_images(item, self.image_col, self.clip_proc)
            if pixel_values is None:
                continue

            raw_text = item[self.text_col]

            if self.text_subfield:
                pairs = raw_text if isinstance(raw_text, (list, tuple)) else [raw_text]
                convs = _build_conversation_examples(pairs, self.text_subfield)
            else:
                caption = raw_text[0] if isinstance(raw_text, (list, tuple)) else raw_text
                convs   = [(caption, 0)]

            for full_text, answer_char_start in convs:
                tok = _tokenize_with_label_mask(
                    full_text, answer_char_start, self.tokenizer, self.max_len
                )
                if tok is None:
                    continue
                yield {
                    "pixel_values": pixel_values,
                    "num_images":   pixel_values.shape[0],
                    **tok,
                }


# ─── Collate function ─────────────────────────────────────────────────────────

def make_collate_fn():
    """
    Pads pixel_values to max N in the batch.
    Now also collates the labels tensor returned by _tokenize_with_label_mask.
    """
    def collate_fn(batch):
        max_n      = max(item["num_images"] for item in batch)
        _, C, H, W = batch[0]["pixel_values"].shape

        padded = torch.zeros(len(batch), max_n, C, H, W,
                             dtype=batch[0]["pixel_values"].dtype)
        for i, item in enumerate(batch):
            n = item["num_images"]
            padded[i, :n] = item["pixel_values"]

        return {
            "pixel_values":   padded,
            "num_images":     torch.tensor([item["num_images"] for item in batch],
                                           dtype=torch.long),
            "input_ids":      torch.stack([item["input_ids"]      for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels":         torch.stack([item["labels"]         for item in batch]),
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

    cfg = _resolve_config(args["dataset"], args.get("dataset_config"))
    load_kwargs = {"split": args["dataset_split"]}
    if cfg:
        load_kwargs["name"] = cfg

    train_samples = args.get("train_samples")
    collate_fn    = make_collate_fn()

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
            text_subfield=args.get("text_subfield"),
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
            text_subfield=args.get("text_subfield"),
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

    val_split = args.get("val_split")
    val_repo  = args.get("val_dataset")

    if val_split is None and val_repo is None:
        return None

    val_split = val_split or "validation"
    val_repo  = val_repo  or args["dataset"]

    cfg = _resolve_config(val_repo, args.get("val_dataset_config"))
    load_kwargs = {"split": val_split}
    if cfg:
        load_kwargs["name"] = cfg

    val_samples = args.get("val_samples")
    collate_fn  = make_collate_fn()

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
                text_subfield=args.get("text_subfield"),
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
                text_subfield=args.get("text_subfield"),
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