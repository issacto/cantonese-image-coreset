%%writefile CoresetWorker.py

import io
import numpy as np
import ray
import torch
from PIL import Image
from typing import List, Optional, Tuple


@ray.remote
class CoresetWorker:
    def __init__(
        self,
        worker_id: int,
        coreset_size: int,
        model_name: str,
        embed_batch_size: int,
    ):
        self.worker_id = worker_id
        self.coreset_size = coreset_size
        self.embed_batch_size = embed_batch_size

        # ── CLIP model (owned by this worker, lives on its 0.5-GPU slice) ─────
        self.device = torch.device("cuda")
        from transformers import CLIPModel, CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # ── Coreset state ──────────────────────────────────────────────────────
        self.coreset_embeddings: Optional[np.ndarray] = None
        self.coreset_indices: Optional[np.ndarray] = None
        self.batches_processed = 0
        self.images_skipped = 0

    def _log(self, msg: str) -> None:
        print(f"[Worker {self.worker_id}] {msg}")

    # ── Embedding (runs inline, no remote hop needed) ─────────────────────────

    @torch.no_grad()
    def _embed(
        self,
        images: List[Optional[bytes]],
    ) -> Tuple[np.ndarray, List[int]]:
        """Convert raw bytes → L2-normalised CLIP embeddings.

        Returns
        -------
        embs        : float32 ndarray of shape (N_valid, D)
        valid_pos   : list of original positions that survived
        """
        pil_images: List[Optional[Image.Image]] = []
        for b in images:
            if b is None:
                pil_images.append(None)
            else:
                try:
                    pil_images.append(Image.open(io.BytesIO(b)).convert("RGB"))
                except Exception as exc:
                    self._log(f"decode error, skipping: {exc}")
                    pil_images.append(None)

        indexed = [(i, img) for i, img in enumerate(pil_images) if img is not None]
        if not indexed:
            return np.empty((0, 0), dtype=np.float32), []

        positions = [pos for pos, _ in indexed]
        valid_imgs = [img for _, img in indexed]

        all_embs: List[np.ndarray] = []
        valid_positions_out: List[int] = []

        for start in range(0, len(valid_imgs), self.embed_batch_size):
            mini_imgs = valid_imgs[start : start + self.embed_batch_size]
            mini_pos  = positions[start : start + self.embed_batch_size]
            try:
                inputs = self.processor(images=mini_imgs, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                out = self.model.vision_model(pixel_values=inputs["pixel_values"])
                feats = self.model.visual_projection(out.pooler_output)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embs.append(feats.cpu().float().numpy())
                valid_positions_out.extend(mini_pos)
            except Exception as exc:
                self._log(f"mini-batch failed ({exc}), retrying one-by-one")
                for img, pos in zip(mini_imgs, mini_pos):
                    try:
                        inputs = self.processor(images=[img], return_tensors="pt", padding=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        out = self.model.vision_model(pixel_values=inputs["pixel_values"])
                        feat = self.model.visual_projection(out.pooler_output)
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                        all_embs.append(feat.cpu().float().numpy())
                        valid_positions_out.append(pos)
                    except Exception as inner_exc:
                        self._log(f"skipping image at pos {pos}: {inner_exc}")

        if not all_embs:
            return np.empty((0, 0), dtype=np.float32), []

        return np.concatenate(all_embs, axis=0), valid_positions_out

    # ── Main entry point ──────────────────────────────────────────────────────

    def process_batch(self, images: List[Optional[bytes]], ids: List[int]) -> dict:
        from coreset.greedy import incremental_coreset_merge

        skipped = sum(1 for img in images if img is None)
        self.images_skipped += skipped

        self._log(f"embedding {len(images)} images ...")
        new_embs, valid_pos = self._embed(images)
        self._log(f"embed done, got {len(new_embs)} embeddings")

        if len(new_embs) == 0:
            self._log(f"batch {self.batches_processed}: all images failed, skipping")
            return {"worker_id": self.worker_id, "status": "empty_batch"}

        valid_ids = np.array([ids[p] for p in valid_pos], dtype=np.int64)

        # ── CPU work: greedy k-center coreset merge ───────────────────────────
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
            "coreset_size": (
                len(self.coreset_embeddings) if self.coreset_embeddings is not None else 0
            ),
            "images_skipped": self.images_skipped,
        }