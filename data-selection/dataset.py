"""
HuggingFace streaming image dataset.

Stopping logic
──────────────
The program covers exactly `total_samples` stream positions, guaranteed:

    page_size × n_pages = total_samples

    e.g. page_size=10,000 × 10 pages = 100,000 total_samples

`sample_size` and `batch_size` are irrelevant to how much of the stream is
covered — they only control shuffle quality and worker throughput respectively.

Each page:
    1. Stream past `page_size` positions  (touches exactly page_size examples)
    2. Keep every stride-th one           (stride = page_size // sample_size)
    3. Shuffle kept examples locally
    4. Serve chunks of `batch_size` from the kept examples

When `_pages_issued == n_pages` the dataset marks itself exhausted and
next_chunk() returns empty lists, which propagates up to the dispatcher.
"""

import random
from typing import Iterator, List, Optional, Tuple

from PIL import Image


class HFStreamingDataset:
    """
    Wraps a HuggingFace IterableDataset for lazy, strided image streaming.

    Args:
        dataset_name:   HF repo id, e.g. "laion/laion2B-en" or a local path.
        split:          Dataset split string, e.g. "train".
        image_col:      Name of the column that holds the image.
        page_size:      Number of stream positions to advance past per refill.
                        Must divide evenly into total_samples.
        sample_size:    Number of examples to keep per page.
                        stride = page_size // sample_size.
                        Must be <= page_size.
        total_samples:  Total stream positions to cover.
                        n_pages = total_samples // page_size.
        seed:           RNG seed for local page shuffle.
        hf_kwargs:      Extra keyword arguments forwarded to `load_dataset`.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_col: str = "image",
        page_size: int = 10_000,
        sample_size: int = 1_000,
        total_samples: int = 100_000,
        seed: int = 42,
        **hf_kwargs,
    ):
        from datasets import load_dataset

        assert sample_size <= page_size, "sample_size must be <= page_size"
        assert total_samples >= page_size, "total_samples must be >= page_size"

        self.image_col    = image_col
        self._page_size   = page_size
        self._sample_size = sample_size
        self._stride      = page_size // sample_size    # e.g. 10000//1000 = 10
        self._n_pages     = total_samples // page_size  # e.g. 100000//10000 = 10
        self._pages_issued = 0

        random.seed(seed)

        ds = load_dataset(
            dataset_name,
            "images",
            split=split,
            streaming=True,
            **hf_kwargs,
        )

        # Cast the image column so HF decodes bytes → PIL automatically
        try:
            from datasets import Image as HFImage
            if image_col in ds.features and isinstance(ds.features[image_col], HFImage):
                ds = ds.cast_column(image_col, HFImage(decode=True))
        except Exception:
            pass

        self._ds = ds
        self._iter: Optional[Iterator] = None
        self._page: List = []
        self._cursor: int = 0
        self._exhausted: bool = False

        print(
            f"[HFStreamingDataset] '{dataset_name}' split='{split}' "
            f"image_col='{image_col}' page_size={page_size} "
            f"sample_size={sample_size} stride={self._stride} "
            f"n_pages={self._n_pages} total_samples={total_samples}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_iter(self) -> Iterator:
        if self._iter is None:
            self._iter = iter(self._ds)
        return self._iter

    def _refill_page(self) -> None:
        """
        Stream past exactly page_size positions, keeping every stride-th one.
        Skipped examples cost only a raw next() — no image decode, no download.
        Kept examples are shuffled locally before being stored in self._page.

        Stops once n_pages have been issued, guaranteeing:
            page_size × n_pages = total_samples stream positions covered.
        """
        if self._pages_issued >= self._n_pages:
            self._exhausted = True
            return

        it = self._get_iter()
        kept = []
        fetched = 0

        while fetched < self._page_size:
            try:
                example = next(it)      # keep this one
                kept.append(example)
                fetched += 1
                # skip (stride-1) examples — raw next(), no decode
                for _ in range(self._stride - 1):
                    next(it)
                    fetched += 1
            except StopIteration:
                # HF stream physically exhausted before n_pages reached
                self._exhausted = True
                break

        random.shuffle(kept)
        self._page = kept
        self._pages_issued += 1

        # mark exhausted after the final page
        if self._pages_issued >= self._n_pages:
            self._exhausted = True

    def _decode_image(self, raw) -> Optional[Image.Image]:
        """
        Accept the many shapes HF datasets give you for image columns:
          - Already a PIL Image                  → use directly
          - dict {"bytes": b"...", "path": ...}  → decode from bytes
          - str / bytes (URL or raw bytes)        → fetch / decode
        Returns None on any failure so the caller can skip silently.
        """
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

    def next_chunk(
        self, chunk_size: int
    ) -> Tuple[List[Optional[bytes]], List[int]]:
        """
        Pull the next `chunk_size` examples from the current page.
        Refills the page automatically when it runs dry.

        Returns:
            images:   list of PNG bytes (or None for failed decodes).
            ids:      list of monotonically increasing stream positions.

        Returns empty lists once all n_pages have been issued.
        """
        images: List[Optional[bytes]] = []
        ids: List[int] = []

        while len(images) < chunk_size:
            if not self._page:
                if self._exhausted:
                    break
                self._refill_page()
                if not self._page:
                    break

            need = chunk_size - len(images)
            batch, self._page = self._page[:need], self._page[need:]

            for example in batch:
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

        return images, ids

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-create the iterator from the start."""
        if seed is not None:
            random.seed(seed)
        self._iter = None
        self._page = []
        self._cursor = 0
        self._pages_issued = 0
        self._exhausted = False