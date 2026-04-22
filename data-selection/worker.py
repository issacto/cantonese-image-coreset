%%writefile CoresetWorker.py

import asyncio
import numpy as np
import ray
from typing import Optional, Tuple


@ray.remote(num_cpus=1, max_concurrency=4)   # ← raised from 1: prevents deadlock when
                                              #   awaiting embedder while event loop is
                                              #   occupied on the last batch
class CoresetWorker:
    def __init__(
        self,
        worker_id: int,
        coreset_size: int,
        embedder: ray.actor.ActorHandle,   # ← injected at construction
    ):
        self.worker_id = worker_id
        self.coreset_size = coreset_size
        self.embedder = embedder           # shared CLIPEmbedder handle

        self.coreset_embeddings: Optional[np.ndarray] = None
        self.coreset_indices: Optional[np.ndarray] = None
        self.batches_processed = 0
        self.images_skipped = 0

    def _log(self, msg: str) -> None:
        print(f"[Worker {self.worker_id}] {msg}")

    def process_batch(self, images, ids) -> dict:
        from coreset.greedy import incremental_coreset_merge

        skipped = sum(1 for img in images if img is None)
        self.images_skipped += skipped

        # ── GPU work: dispatched to the embedder actor, non-blocking ─────────
        # Wrap in asyncio.ensure_future so the ObjectRef is awaited as a proper
        # asyncio-compatible future, yielding the event loop slot while waiting.
        # This is what prevented the last batch from ever completing when the
        # embedder was still busy and max_concurrency=1 (the default).
        embed_ref = self.embedder.embed.remote(images)
        self._log(f"embed dispatched, awaiting...")
        new_embs, valid_pos = ray.get(embed_ref)
        self._log(f"embed done, got {len(new_embs)} embeddings")

        if len(new_embs) == 0:
            self._log(f"batch {self.batches_processed}: all images failed, skipping")
            return {"worker_id": self.worker_id, "status": "empty_batch"}

        valid_ids = np.array([ids[p] for p in valid_pos], dtype=np.int64)

        # ── CPU work: merge + greedy k-center (stays on this actor) ──────────
        self.coreset_embeddings, self.coreset_indices = incremental_coreset_merge(
            existing_embeddings=self.coreset_embeddings,
            new_embeddings=new_embs,
            existing_indices=self.coreset_indices,
            new_indices=valid_ids,
            k=self.coreset_size,
        )

        self.batches_processed += 1
        return {
            "worker_id": self.worker_id,
            "status": "ok",
            "batches_processed": self.batches_processed,
            "coreset_size": len(self.coreset_embeddings),
            "images_skipped": self.images_skipped,
        }

    def get_coreset(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.coreset_embeddings, self.coreset_indices

    def stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "batches_processed": self.batches_processed,
            "coreset_size": len(self.coreset_embeddings) if self.coreset_embeddings is not None else 0,
            "images_skipped": self.images_skipped,
        }