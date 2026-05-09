"""
CoresetWorker — self-contained shard-per-worker edition.

Each worker:
    1. Owns its own HFStreamingDataset shard (no dispatcher needed)
    2. Streams page_size examples, keeps sample_size randomly
    3. Embeds kept examples with CLIP
    4. Runs greedy k-center on the page embeddings
    5. Merges result into a running local coreset
    6. Repeats until n_pages exhausted
    7. Returns (embeddings, ids) to the driver for global merge
"""

import io
from typing import List, Optional, Tuple

import numpy as np
import ray
import torch
import os
from huggingface_hub import login

@ray.remote
class CoresetWorker:
    """
    Self-contained coreset worker. Owns one dataset shard and runs the
    full load → embed → greedy pipeline independently.

    Args:
        worker_id:          Index of this worker (= shard index).
        num_workers:        Total number of workers (= num_shards).
        dataset_name:       HF repo id.
        split:              Dataset split.
        image_col:          Image column name.
        page_size:          Stream positions per page refill.
        sample_size:        Examples kept per page.
        total_samples:      Total stream positions across ALL workers.
                            This worker covers total_samples // num_workers.
        coreset_size:       Max local coreset size kept between pages.
        model_name:         HF CLIP model name.
        embed_batch_size:   Images per CLIP forward pass.
        seed:               RNG seed.
        hf_kwargs:          Extra kwargs for load_dataset.
    """

    def __init__(
        self,
        worker_id: int,
        num_workers: int,
        dataset_name: str,
        split: str,
        image_col: str,
        page_size: int,
        sample_size: int,
        total_samples: int,
        coreset_size: int,
        model_name: str,
        embed_batch_size: int,
        seed: int,
        token=None,
        **hf_kwargs,
    ):
        hf_token = token or os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        from coreset.dataset import HFStreamingDataset

        self.worker_id       = worker_id
        self.coreset_size    = coreset_size
        self.embed_batch_size = embed_batch_size

        samples_per_worker = total_samples // num_workers

        self._ds = HFStreamingDataset(
            dataset_name=dataset_name,
            split=split,
            image_col=image_col,
            num_shards=num_workers,
            shard_index=worker_id,
            page_size=page_size,
            sample_size=sample_size,
            samples_per_worker=samples_per_worker,
            seed=seed,
            **hf_kwargs,
        )

        # CLIP model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model(model_name)

        # Local coreset state
        self._coreset_embs: Optional[np.ndarray] = None
        self._coreset_ids:  Optional[np.ndarray] = None

        # Stats
        self._pages_processed = 0
        self._images_skipped  = 0

        print(
            f"[Worker {worker_id}] ready — device={self._device} "
            f"shard={worker_id}/{num_workers} "
            f"samples_per_worker={samples_per_worker} "
            f"n_pages={self._ds.n_pages}"
        )

    # ── Model ─────────────────────────────────────────────────────────────────

    def _load_model(self, model_name: str) -> None:
        from transformers import CLIPModel, CLIPProcessor
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, image_bytes_list):
        from PIL import Image

        pil_images, valid_indices = [], []
        for i, img_bytes in enumerate(image_bytes_list):
            if img_bytes is None:
                self._images_skipped += 1
                continue
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                pil_images.append(img)
                valid_indices.append(i)
            except Exception:
                self._images_skipped += 1

        if not pil_images:
            return np.empty((0, 0)), []

        all_embs = []
        for i in range(0, len(pil_images), self.embed_batch_size):
            batch = pil_images[i: i + self.embed_batch_size]
            with torch.no_grad():
                inputs = self._processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                # ✅ Same pattern as your working ClipWorker
                out = self._model.vision_model(pixel_values=inputs["pixel_values"])
                feats = self._model.visual_projection(out.pooler_output)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embs.append(feats.cpu().numpy())

        return np.concatenate(all_embs, axis=0), valid_indices
    # ── Greedy k-center ───────────────────────────────────────────────────────

    def _greedy_kcenter(
        self, embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Greedy k-center on a set of L2-normalized embeddings.
        Returns (selected_embeddings, selected_indices).
        """
        from coreset.greedy import greedy_kcenter
        sel_indices, _ = greedy_kcenter(embeddings, k)
        return embeddings[sel_indices], sel_indices

    # ── Local coreset merge ───────────────────────────────────────────────────

    def _merge_into_coreset(
        self, new_embs: np.ndarray, new_ids: np.ndarray
    ) -> None:
        """
        Pool current local coreset with new page embeddings,
        then run greedy k-center to keep at most coreset_size points.
        """
        if self._coreset_embs is not None and len(self._coreset_embs) > 0:
            pool_embs = np.concatenate([self._coreset_embs, new_embs], axis=0)
            pool_ids  = np.concatenate([self._coreset_ids,  new_ids],  axis=0)
        else:
            pool_embs = new_embs
            pool_ids  = new_ids

        k = min(self.coreset_size, len(pool_embs))
        sel_embs, sel_idx = self._greedy_kcenter(pool_embs, k)

        self._coreset_embs = sel_embs
        self._coreset_ids  = pool_ids[sel_idx]

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the full pipeline for this worker's shard:
            for each page:
                1. Load page from shard
                2. Embed page images with CLIP
                3. Run greedy k-center on page embeddings
                4. Merge into local coreset

        Returns:
            (coreset_embeddings, coreset_ids)
        """
        print(f"[Worker {self.worker_id}] starting — {self._ds.n_pages} pages to process")

        while not self._ds.is_exhausted:
            images, ids = self._ds.next_page()

            if not images:
                break

            # Embed
            embs, valid_idx = self._embed(images)

            if len(embs) == 0:
                print(f"[Worker {self.worker_id}] page {self._pages_processed} — all images failed decode, skipping")
                self._pages_processed += 1
                continue

            # Map ids to only the valid (decoded) ones
            ids_arr = np.array(ids)
            valid_ids = ids_arr[valid_idx]

            # Merge into local coreset
            self._merge_into_coreset(embs, valid_ids)
            self._pages_processed += 1

            print(
                f"[Worker {self.worker_id}] page {self._pages_processed}/{self._ds.n_pages} done — "
                f"local coreset size: {len(self._coreset_embs)}"
            )

        print(
            f"[Worker {self.worker_id}] finished — "
            f"pages={self._pages_processed} "
            f"coreset_size={len(self._coreset_embs) if self._coreset_embs is not None else 0} "
            f"skipped={self._images_skipped}"
        )

        return self._coreset_embs, self._coreset_ids

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "worker_id":       self.worker_id,
            "pages_processed": self._pages_processed,
            "coreset_size":    len(self._coreset_embs) if self._coreset_embs is not None else 0,
            "images_skipped":  self._images_skipped,
        }
