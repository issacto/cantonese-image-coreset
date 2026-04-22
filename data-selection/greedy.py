%%writefile greedy.py
"""
Greedy coreset selection.

Algorithm: Greedy k-Center (farthest-point sampling).
  - Iteratively selects the point farthest from the current selection.
  - Gives a 2-approximation to the optimal k-center objective.
  - Assumes embeddings are L2-normalised so cosine distance = 1 - dot(a, b).

Complexity: O(N * k) time, O(N) space.
"""

import numpy as np
from typing import Optional, Tuple


def greedy_kcenter(
    embeddings: np.ndarray,
    k: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select k diverse points from `embeddings` via greedy farthest-point sampling.

    Args:
        embeddings:  (N, D) float32, L2-normalised unit vectors.
        k:           Number of points to select.  Clamped to N if k > N.
        seed:        Optional fixed starting index (for reproducibility).
                     If None, a random point is chosen.

    Returns:
        selected:    (k,) int64 — indices into `embeddings`.
        min_dists:   (N,) float32 — min cosine distance from each point to its
                     nearest selected neighbour (useful for weighted sampling).
    """
    N = len(embeddings)
    k = min(k, N)

    # -inf marks already-selected points so argmax skips them
    min_dists = np.full(N, np.inf, dtype=np.float32)
    selected: List[int] = []

    # Seed point
    first = seed if seed is not None else int(np.random.randint(N))
    _add_point(first, embeddings, selected, min_dists)

    while len(selected) < k:
        nxt = int(np.argmax(min_dists))
        _add_point(nxt, embeddings, selected, min_dists)

    return np.array(selected, dtype=np.int64), min_dists.clip(min=0)


def _add_point(
    idx: int,
    embeddings: np.ndarray,
    selected: list,
    min_dists: np.ndarray,
) -> None:
    """Select `idx`, update min_dists, mark it as taken."""
    selected.append(idx)
    # Cosine distance to every other point (vectorised dot-product)
    dists = 1.0 - (embeddings @ embeddings[idx]).astype(np.float32)
    np.minimum(min_dists, dists, out=min_dists)
    min_dists[idx] = -np.inf  # mark as selected; argmax will never pick it again


# ── Incremental merge ─────────────────────────────────────────────────────────

def incremental_coreset_merge(
    existing_embeddings: Optional[np.ndarray],
    new_embeddings: np.ndarray,
    existing_indices: Optional[np.ndarray],
    new_indices: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge an existing coreset with a fresh batch of embeddings, then prune
    the combined pool back to k points via greedy k-center.

    By running greedy on the full combined pool (not just the new points),
    we allow existing coreset members to be replaced if a new point covers
    their neighbourhood better — giving an honest approximation guarantee.

    Args:
        existing_embeddings: (M, D) or None on the first call.
        new_embeddings:      (B, D) float32, L2-normalised.
        existing_indices:    (M,) dataset indices matching existing_embeddings.
        new_indices:         (B,) dataset indices matching new_embeddings.
        k:                   Target coreset size after merge.

    Returns:
        selected_embeddings: (min(k, M+B), D)
        selected_indices:    (min(k, M+B),)
    """
    if existing_embeddings is None or len(existing_embeddings) == 0:
        pool_embs = new_embeddings
        pool_idx = new_indices
    else:
        pool_embs = np.concatenate([existing_embeddings, new_embeddings], axis=0)
        pool_idx = np.concatenate([existing_indices, new_indices], axis=0)

    sel, _ = greedy_kcenter(pool_embs, k)
    return pool_embs[sel], pool_idx[sel]


# ── Type hint shim (List used in _add_point) ─────────────────────────────────
from typing import List  # noqa: E402 — intentional bottom import
