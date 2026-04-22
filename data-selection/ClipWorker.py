%%writefile ClipWorker.py

import io
import numpy as np
import torch
import ray
from PIL import Image
from typing import List, Optional, Tuple


@ray.remote(num_gpus=1)
class ClipWorker:
    def __init__(self, model_name: str, embed_batch_size: int):
        self.embed_batch_size = embed_batch_size
        self.device = torch.device("cuda")

        from transformers import CLIPModel, CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _get_feats(self, inputs):
        print("[ClipWorker] running vision_model...")
        out = self.model.vision_model(pixel_values=inputs["pixel_values"])
        print("[ClipWorker] vision_model done")
        feats = self.model.visual_projection(out.pooler_output)
        return feats / feats.norm(dim=-1, keepdim=True)


    @torch.no_grad()
    def embed(
        self,
        images: List[Optional[bytes]],
    ) -> Tuple[np.ndarray, List[int]]:
        print(len(images), "images")

        pil_images = []
        for b in images:
            if b is None:
                pil_images.append(None)
            else:
                pil_images.append(Image.open(io.BytesIO(b)).convert("RGB"))

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
                feats = self._get_feats(inputs)
                all_embs.append(feats.cpu().float().numpy())
                valid_positions_out.extend(mini_pos)
            except Exception as exc:
                print(f"[ClipWorker] mini-batch failed ({exc}), retrying one-by-one")
                for img, pos in zip(mini_imgs, mini_pos):
                    try:
                        inputs = self.processor(images=[img], return_tensors="pt", padding=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        feat = self._get_feats(inputs)
                        all_embs.append(feat.cpu().float().numpy())
                        valid_positions_out.append(pos)
                    except Exception as inner_exc:
                        print(f"[ClipWorker] skipping image at pos {pos}: {inner_exc}")

        if not all_embs:
            return np.empty((0, 0), dtype=np.float32), []

        return np.concatenate(all_embs, axis=0), valid_positions_out