"""
Coreset worker Ray actor — HuggingFace streaming edition.

What changed vs. the file-based version
─────────────────────────────────────────
- Workers no longer own a dataset / sampler.  Images arrive pre-fetched
  from the dispatcher via Ray object-store references.
- `process_batch(images_ref, ids_ref)` receives Ray ObjectRefs instead of
  an index array; it calls ray.get() locally to materialise the data.
- Everything else (CLIP embedding, greedy k-center merge) is unchanged.

Each worker:
  1. Holds one CLIP model instance (loaded once at startup).
  2. Maintains a rolling local coreset of `coreset_size` embeddings.
  3. When given a batch:
       a. Materialises pre-fetched PIL images from the Ray object store.
       b. Embeds them in mini-batches of `embed_batch_size` via CLIP.
       c. Merges the new embeddings with its local coreset.
       d. Prunes the merged pool back to `coreset_size` via greedy k-center.

Fractional GPU allocation is handled by Ray via .options(num_gpus=0.5);
two workers share one physical GPU transparently.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import ray
from PIL import Image


@ray.remote
class CorsetWorker:
    def __init__(
        self,
        worker_id: int,
        model_name: str,
        coreset_size: int,
        embed_batch_size: int,
    ):
        self.worker_id = worker_id
        self.coreset_size = coreset_size
        self.embed_batch_size = embed_batch_size

        # ── Device ────────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log(f"device={self.device}")

        # ── CLIP model ────────────────────────────────────────────────────────
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._log(f"model loaded: {model_name}")

        # ── Local coreset state ───────────────────────────────────────────────
        self.coreset_embeddings: Optional[np.ndarray] = None  # (coreset_size, D)
        self.coreset_indices: Optional[np.ndarray] = None     # (coreset_size,)
        self.batches_processed: int = 0
        self.images_skipped: int = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        print(f"[Worker {self.worker_id}] {msg}")

    @torch.no_grad()
    def _embed(
        self, images: List[Optional[Image.Image]]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Embed a list of images (None entries are skipped) in mini-batches.

        Returns:
            embeddings:      (V, D) float32, L2-normalised.
            valid_positions: list of length V — maps row i of embeddings
                             back to its position in the original `images` list.
        """
        indexed = [(i, img) for i, img in enumerate(images) if img is not None]
        if not indexed:
            return np.empty((0, 0), dtype=np.float32), []

        positions = [pos for pos, _ in indexed]
        valid_imgs = [img for _, img in indexed]

        all_embs: List[np.ndarray] = []
        for start in range(0, len(valid_imgs), self.embed_batch_size):
            mini = valid_imgs[start : start + self.embed_batch_size]
            inputs = self.processor(images=mini, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_image_features(**inputs)
            # unwrap HF output if needed
            if not torch.is_tensor(feats):
                feats = feats.pooler_output

            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embs.append(feats.cpu().float().numpy())

        return np.concatenate(all_embs, axis=0), positions

    # ── Public Ray methods ────────────────────────────────────────────────────

    def process_batch(
        self,
        images,
        ids
    ) -> dict:
        """
        Called by the driver for each pre-fetched batch from the dispatcher.

        Args:
            images_ref: Ray ObjectRef → List[Optional[PIL.Image]]
                        Pre-fetched and shuffled by the dispatcher.
            ids_ref:    Ray ObjectRef → List[int]
                        Monotonic stream positions used as stable coreset IDs.

        Steps:
          1. Materialise images from Ray object store (zero-copy shared memory).
          2. Embed valid images in mini-batches of embed_batch_size.
          3. Merge new embeddings with local coreset.
          4. Prune combined pool back to coreset_size via greedy k-center.
        """
        from coreset.greedy import incremental_coreset_merge

        # 1. Materialise from Ray object store
        # images: List[Optional[Image.Image]] = ray.get(images_ref)
        # ids: List[int] = ray.get(ids_ref)

        skipped = sum(1 for img in images if img is None)
        self.images_skipped += skipped

        # 2. Embed
        new_embs, valid_pos = self._embed(images)

        if len(new_embs) == 0:
            self._log(
                f"batch {self.batches_processed}: all images failed to decode, skipping"
            )
            return {"worker_id": self.worker_id, "status": "empty_batch"}

        # Map valid positions back to their stream IDs
        valid_ids = np.array([ids[p] for p in valid_pos], dtype=np.int64)

        # 3 & 4. Merge + prune
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
        """Return (embeddings, stream_ids) for this worker's local coreset."""
        return self.coreset_embeddings, self.coreset_indices

    def stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "batches_processed": self.batches_processed,
            "coreset_size": (
                len(self.coreset_embeddings) if self.coreset_embeddings is not None else 0
            ),
            "images_skipped": self.images_skipped,
        }
