"""
HuggingFace streaming image dataset — shard-per-worker edition.

Each worker owns exactly one shard of the dataset:
    ds = load_dataset(..., streaming=True)
    ds = ds.shard(num_shards=num_workers, index=worker_id)

This means network traffic is fully partitioned — worker 0 only downloads
shard 0's parquet files, worker 1 only downloads shard 1's, etc.
(Assumes dataset has >= num_workers underlying parquet files.)

Stopping logic
──────────────
    page_size × n_pages = samples_per_worker
    samples_per_worker  = total_samples // num_workers

Each worker covers exactly samples_per_worker stream positions from its shard.
Page-based strided sampling:
    1. Stream past page_size positions
    2. Keep every stride-th one  (stride = page_size // sample_size)
    3. Shuffle kept examples locally
    4. Caller runs greedy k-center on kept examples
"""

import random
from typing import Iterator, List, Optional, Tuple

from PIL import Image


class HFStreamingDataset:
    """
    One shard of a HuggingFace IterableDataset for a single worker.

    Args:
        dataset_name:       HF repo id, e.g. "laion/laion2B-en".
        split:              Dataset split, e.g. "train".
        image_col:          Column name that holds the image.
        num_shards:         Total number of workers (= total shards).
        shard_index:        This worker's shard index (0-based).
        page_size:          Stream positions advanced per page refill.
        sample_size:        Examples kept per page (stride = page_size // sample_size).
        samples_per_worker: Total stream positions this worker covers.
                            = total_samples // num_workers
        seed:               RNG seed for local page shuffle.
        hf_kwargs:          Extra kwargs forwarded to load_dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_col: str = "image",
        num_shards: int = 1,
        shard_index: int = 0,
        page_size: int = 2_000,
        sample_size: int = 256,
        samples_per_worker: int = 10_000,
        seed: int = 42,
        **hf_kwargs,
    ):
        from datasets import load_dataset

        assert sample_size <= page_size, "sample_size must be <= page_size"
        assert samples_per_worker >= page_size, "samples_per_worker must be >= page_size"

        self.image_col          = image_col
        self._page_size         = page_size
        self._sample_size       = sample_size
        self._stride            = page_size // sample_size
        self._n_pages           = samples_per_worker // page_size
        self._pages_issued      = 0

        random.seed(seed + shard_index)   # different seed per worker

        ds = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
            **hf_kwargs,
        )

        # Partition: worker only sees its own slice of parquet files
        ds = ds.shard(num_shards=num_shards, index=shard_index)

        # Cast image column for auto PIL decode if applicable
        try:
            from datasets import Image as HFImage
            if image_col in ds.features and isinstance(ds.features[image_col], HFImage):
                ds = ds.cast_column(image_col, HFImage(decode=True))
        except Exception:
            pass

        self._ds            = ds
        self._iter: Optional[Iterator] = None
        self._page: List    = []
        self._cursor: int   = 0
        self._exhausted: bool = False

        print(
            f"[HFStreamingDataset] worker={shard_index}/{num_shards} "
            f"dataset='{dataset_name}' split='{split}' "
            f"page_size={page_size} sample_size={sample_size} "
            f"stride={self._stride} n_pages={self._n_pages} "
            f"samples_per_worker={samples_per_worker}"
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_iter(self) -> Iterator:
        if self._iter is None:
            self._iter = iter(self._ds)
        return self._iter

    def _refill_page(self) -> None:
        """
        Stream past exactly page_size positions from this worker's shard,
        keeping every stride-th example. Kept examples are locally shuffled.
        Stops once n_pages have been issued.
        """
        if self._pages_issued >= self._n_pages:
            self._exhausted = True
            return

        it = self._get_iter()
        kept = []
        fetched = 0

        while fetched < self._page_size:
            try:
                example = next(it)          # keep this one
                kept.append(example)
                fetched += 1
                for _ in range(self._stride - 1):
                    next(it)                # skip — raw next(), no decode
                    fetched += 1
            except StopIteration:
                self._exhausted = True      # shard exhausted before n_pages
                break

        random.shuffle(kept)
        self._page = kept
        self._pages_issued += 1

        if self._pages_issued >= self._n_pages:
            self._exhausted = True

    def _decode_image(self, raw) -> Optional[Image.Image]:
        try:
            if isinstance(raw, Image.Image):
                return raw.convert("RGB")
            if isinstance(raw, dict):
                img_bytes = raw.get("bytes")
                if img_bytes:
                    import io
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                path = raw.get("path")
                if path:
                    return Image.open(path).convert("RGB")
                return None
            if isinstance(raw, (bytes, bytearray)):
                import io
                return Image.open(io.BytesIO(raw)).convert("RGB")
            if isinstance(raw, str):
                import io, urllib.request
                with urllib.request.urlopen(raw, timeout=10) as resp:
                    return Image.open(io.BytesIO(resp.read())).convert("RGB")
        except Exception as e:
            print(f"[decode] failed: {type(raw)} {e}")
            return None
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def next_page(self) -> Tuple[List[Optional[bytes]], List[int]]:
        """
        Return the next full page of kept+shuffled examples.
        The caller (CoresetWorker) runs greedy on the whole page at once.

        Returns:
            images:  list of PNG bytes (or None for decode failures).
            ids:     monotonically increasing stream positions.

        Returns empty lists when all n_pages have been issued.
        """
        if self._exhausted and not self._page:
            return [], []

        if not self._page:
            self._refill_page()
            if not self._page:
                return [], []

        images: List[Optional[bytes]] = []
        ids: List[int] = []

        for example in self._page:
            raw = example.get(self.image_col)

            if not raw or (isinstance(raw, list) and len(raw) == 0):
                images.append(None)
                ids.append(self._cursor)
                self._cursor += 1
                continue

            raw = raw[0] if isinstance(raw, list) else raw
            img = self._decode_image(raw)

            if img is not None:
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                images.append(buf.getvalue())
            else:
                images.append(None)

            ids.append(self._cursor)
            self._cursor += 1

        self._page = []   # page consumed
        return images, ids

    @property
    def pages_issued(self) -> int:
        return self._pages_issued

    @property
    def n_pages(self) -> int:
        return self._n_pages

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted
