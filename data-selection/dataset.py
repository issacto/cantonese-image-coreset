%%writefile dataset.py

"""
HuggingFace streaming image dataset.

Key design decisions for a 4 TB remote dataset
───────────────────────────────────────────────
1. `streaming=True`  — never materialises the dataset locally; images are
   fetched on-demand from the HF hub (or S3 mirror) via HTTP range requests.

2. `.shuffle(buffer_size, seed)`  — HF's built-in streaming shuffle fills a
   RAM buffer of `buffer_size` decoded examples, then samples from it uniformly.
   This gives a good approximation of a global shuffle at O(buffer_size) RAM
   cost rather than O(dataset_size).  Typical good value: 10 000 – 50 000.

3. Lazy decoding  — each item is decoded only when pulled from the iterator.
   The PIL image is returned and immediately discarded by the caller after
   embedding; nothing accumulates in memory.

4. Fault tolerance  — corrupt / undecodable shards produce None; callers skip
   those slots without crashing.
"""

from typing import Iterator, List, Optional, Tuple

from PIL import Image


class HFStreamingDataset:
    """
    Wraps a HuggingFace IterableDataset for lazy, shuffled image streaming.

    Args:
        dataset_name:   HF repo id, e.g. "laion/laion2B-en" or a local path.
        split:          Dataset split string, e.g. "train".
        image_col:      Name of the column that holds the image.
                        - If the column dtype is Image (already decoded PIL),
                          it is used directly.
                        - If the column dtype is dict with a "bytes" key
                          (raw bytes), it is decoded here.
                        - If the column holds a URL string, it is fetched.
        shuffle_buffer: Number of examples kept in the shuffle reservoir.
                        Larger = better randomness, more RAM.
                        Set to 0 or None to disable shuffling (not recommended).
        seed:           RNG seed passed to the HF shuffle.
        hf_kwargs:      Extra keyword arguments forwarded to `load_dataset`
                        (e.g. `use_auth_token`, `data_files`, `name`).
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_col: str = "image",
        shuffle_buffer: int = 10_000,
        seed: int = 42,
        **hf_kwargs,
    ):
        from datasets import load_dataset

        self.image_col = image_col

        ds = load_dataset(
            dataset_name,
            "images",
            split=split,
            streaming=True,
            **hf_kwargs,
        )

        if shuffle_buffer:
            ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)

        # Cast the image column so HF decodes bytes → PIL automatically
        # (only applies when the column has the Image feature type)
        try:
            from datasets import Image as HFImage
            if image_col in ds.features and isinstance(ds.features[image_col], HFImage):
                ds = ds.cast_column(image_col, HFImage(decode=True))
        except Exception:
            pass  # non-fatal; we handle raw bytes below

        self._ds = ds
        self._iter: Optional[Iterator] = None

        print(
            f"[HFStreamingDataset] '{dataset_name}' split='{split}' "
            f"image_col='{image_col}' shuffle_buffer={shuffle_buffer}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_iter(self) -> Iterator:
        if self._iter is None:
            self._iter = iter(self._ds)
        return self._iter

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
                # Treat as URL — fetch with a short timeout
                import io, urllib.request
                with urllib.request.urlopen(raw, timeout=10) as resp:
                    return Image.open(io.BytesIO(resp.read())).convert("RGB")

        except Exception:
            return None

        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def next_chunk(
        self, chunk_size: int
    ) -> Tuple[List[Optional[Image.Image]], List[int]]:
        """
        Pull the next `chunk_size` examples from the stream.

        Returns:
            images:   list of PIL Images (or None for failed decodes).
            ids:      list of monotonically increasing stream positions
                      (used as stable identifiers in the coreset index).

        When the stream is exhausted the lists may be shorter than chunk_size
        (or empty).  The caller should treat an empty return as "done".
        """
        it = self._get_iter()
        images: List[Optional[Image.Image]] = []
        ids: List[int] = []

        start_id = getattr(self, "_cursor", 0)

        for local_i in range(chunk_size):
            try:
                example = next(it)
            except StopIteration:
                break

            raw_list = example.get(self.image_col)

            if not raw_list or len(raw_list) == 0:
                images.append(None)
                continue

            raw = raw_list[0] if isinstance(raw_list, list) else raw_list
            images.append(self._decode_image(raw))
            ids.append(start_id + local_i)

        self._cursor = start_id + len(images)
        return images, ids

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-create the iterator (new shuffle permutation if seed differs)."""
        if seed is not None:
            self._ds = self._ds.shuffle(buffer_size=self._ds._ex_iterable.ex_iterable.buffer_size, seed=seed)
        self._iter = None
        self._cursor = 0
