"""
Work dispatcher Ray actor — HuggingFace streaming edition.

Stopping logic
──────────────
Stopping is driven entirely by the dataset via page counting:

    page_size × n_pages = total_samples

The dataset marks itself exhausted after n_pages refills. When next_chunk()
returns an empty list, _fetch_one() propagates exhaustion up to get_batch(),
which returns None to workers — signalling the work-stealing loop to stop.

Queue filling
─────────────
A background thread continuously keeps self._queue topped up to MAX_QUEUE
batches. The lock is held ONLY when reading/writing the queue or flags —
never during HF I/O — so get_batch() always returns instantly.
"""

import threading
import time
from typing import Optional, List
import ray


MAX_QUEUE = 16


@ray.remote
class WorkDispatcher:

    def __init__(
        self,
        dataset_name: str,
        split: str,
        image_col: str,
        page_size: int,
        sample_size: int,
        total_samples: int,
        seed: int,
        batch_size: int,
        **hf_kwargs,
    ):
        from coreset.dataset import HFStreamingDataset

        self.batch_size    = batch_size
        self.total_samples = total_samples

        self._ds = HFStreamingDataset(
            dataset_name=dataset_name,
            split=split,
            image_col=image_col,
            page_size=page_size,
            sample_size=sample_size,
            total_samples=total_samples,
            seed=seed,
            **hf_kwargs,
        )

        self._images_issued: int = 0
        self._batches_issued: int = 0
        self._exhausted: bool = False
        self._queue: List[dict] = []

        # Lock guards ONLY queue/flag reads+writes, never I/O
        self._lock = threading.Lock()

        # Pre-fill synchronously before bg thread starts
        print("[WorkDispatcher] Pre-filling batch queue ...")
        for _ in range(8):
            batch = self._do_fetch()
            if batch is None:
                break
            self._queue.append(batch)
        print(f"[WorkDispatcher] Queue ready with {len(self._queue)} batches")

        # Background thread keeps queue topped up
        self._bg = threading.Thread(target=self._background_fill, daemon=True)
        self._bg.start()
        print("[WorkDispatcher] Background fill thread started")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _do_fetch(self) -> Optional[dict]:
        """
        Does the actual HF I/O. NOT thread-safe on _exhausted/_images_issued —
        only call this from the bg thread (after the pre-fill phase).
        """
        images, ids = self._ds.next_chunk(self.batch_size)

        if not images:
            return None

        self._images_issued += len(images)
        self._batches_issued += 1
        return {"images": images, "ids": ids}

    def _background_fill(self) -> None:
        """
        Runs in a daemon thread. Fetches batches from HF and appends to queue.
        Lock is held ONLY for the brief queue append — never during I/O.
        """
        while True:
            # Check queue depth without holding lock for I/O
            with self._lock:
                if self._exhausted:
                    break
                queue_len = len(self._queue)

            if queue_len >= MAX_QUEUE:
                time.sleep(0.05)
                continue

            # Do the slow HF fetch WITHOUT holding the lock
            batch = self._do_fetch()

            if batch is None:
                with self._lock:
                    self._exhausted = True
                break

            # Only hold lock for the brief append
            with self._lock:
                self._queue.append(batch)

    # ── Worker-facing API ─────────────────────────────────────────────────────

    def get_batch(self) -> Optional[dict]:
        """
        Always returns instantly — just pops from the pre-filled queue.
        If queue is temporarily empty, spins briefly waiting for bg thread.
        """
        for _ in range(40):   # wait up to 4 seconds total
            with self._lock:
                if self._queue:
                    batch = self._queue.pop(0)
                    print(
                        f"[Dispatcher] batch #{self._batches_issued}  |  "
                        f"pages covered: {self._ds._pages_issued}/{self._ds._n_pages}  |  "
                        f"images issued so far: {self._images_issued:,}  |  "
                        f"queue depth: {len(self._queue)}"
                    )
                    return batch
                if self._exhausted:
                    return None
            time.sleep(0.1)

        # Timed out — treat as exhausted
        print("[Dispatcher] get_batch timed out waiting for queue")
        return None

    # ── Driver-facing API ─────────────────────────────────────────────────────

    def progress(self) -> dict:
        with self._lock:
            images_issued  = self._images_issued
            batches_issued = self._batches_issued
            exhausted      = self._exhausted
            queue_len      = len(self._queue)

        pct = round(100.0 * images_issued / self.total_samples, 2)
        return {
            "images_issued":  images_issued,
            "total_samples":  self.total_samples,
            "pct":            pct,
            "batches_issued": batches_issued,
            "exhausted":      exhausted,
            "remaining":      max(0, self.total_samples - images_issued),
            "queue_depth":    queue_len,
        }

    def is_exhausted(self) -> bool:
        with self._lock:
            return self._exhausted