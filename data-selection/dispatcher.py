%%writefile dispatcher.py
"""
Work dispatcher Ray actor — HuggingFace streaming edition.

Architecture change vs. the file-based version
───────────────────────────────────────────────
Previously: dispatcher handed out *integer index arrays*; workers loaded
  images themselves from a local directory.

Now: the dataset lives on HuggingFace (4 TB, remote).  We cannot give each
  worker its own independent HF stream — that would multiply network traffic
  by the number of workers and defeat the shuffle buffer (each worker would
  see a different, small portion of the buffer).

Solution — single stream, pre-fetched batches:
  1. The dispatcher owns ONE HFStreamingDataset iterator.
  2. It reads `batch_size` images at a time from that iterator.
  3. Each batch is returned directly as plain Python objects; Ray serializes
     them through its object store automatically when passing to workers.
  4. Workers receive (images, ids) directly — no ray.get() call needed.

Thread-safety: Ray actors execute methods serially — no locks needed.
"""

from typing import Optional, Tuple

import ray


@ray.remote
class WorkDispatcher:
    """
    Streams batches from a HuggingFace dataset and hands them to workers.

    Args:
        dataset_name:    HF repo id (e.g. "laion/laion2B-en").
        split:           Dataset split (e.g. "train").
        image_col:       Column name for images.
        shuffle_buffer:  HF shuffle reservoir size.
        seed:            RNG seed.
        batch_size:      Images per batch handed to each worker.
        total_samples:   Stop after this many images (None = stream forever).
        hf_kwargs:       Forwarded to load_dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        image_col: str,
        shuffle_buffer: int,
        seed: int,
        batch_size: int,
        total_samples: Optional[int] = None,
        **hf_kwargs,
    ):
        from coreset.dataset import HFStreamingDataset

        self.batch_size = batch_size
        self.total_samples = total_samples  # None → unlimited

        self._ds = HFStreamingDataset(
            dataset_name=dataset_name,
            split=split,
            image_col=image_col,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            **hf_kwargs,
        )

        self._images_issued: int = 0
        self._batches_issued: int = 0
        self._exhausted: bool = False

    # ── Worker-facing API ─────────────────────────────────────────────────────

    def get_batch(self) -> Optional[dict]:
        """
        Pull the next batch from the HF stream.

        Returns:
            {"images": List[Optional[PIL.Image]], "ids": List[int]}
            None when the stream is exhausted or total_samples reached.

        Ray serializes the dict through the object store automatically;
        workers receive plain Python objects with no ray.get() needed.
        """
        if self._exhausted:
            return None

        # How many images are we allowed to fetch this round?
        remaining_budget = (
            self.total_samples - self._images_issued
            if self.total_samples is not None
            else self.batch_size
        )
        if remaining_budget <= 0:
            self._exhausted = True
            return None

        fetch_size = min(self.batch_size, remaining_budget)
        images, ids = self._ds.next_chunk(fetch_size)

        if not images:
            self._exhausted = True
            return None

        self._images_issued += len(images)
        self._batches_issued += 1

        if self.total_samples is not None and self._images_issued >= self.total_samples:
            self._exhausted = True

        return {
            "images": images,
            "ids": ids,
        }

    # ── Driver-facing API ─────────────────────────────────────────────────────

    def progress(self) -> dict:
        if self.total_samples:
            pct = 100.0 * self._images_issued / self.total_samples
        else:
            pct = float("nan")
        return {
            "images_issued": self._images_issued,
            "total_samples": self.total_samples,
            "pct": round(pct, 2),
            "batches_issued": self._batches_issued,
            "exhausted": self._exhausted,
            "remaining": (
                max(0, self.total_samples - self._images_issued)
                if self.total_samples
                else None
            ),
        }

    def is_exhausted(self) -> bool:
        return self._exhausted
