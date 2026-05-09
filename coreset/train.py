"""
train.py — Distributed Greedy Coreset Selection (shard-per-worker edition)

Usage example:
    python train.py \
        --hf-dataset HuggingFaceM4/Docmatix \
        --hf-split train \
        --hf-image-col images \
        --hf-text-col texts \
        --total-samples 100000 \
        --page-size 1000 \
        --sample-size 256 \
        --final-coreset-size 10000 \
        --local-coreset-size 1000 \
        --embed-batch-size 128 \
        --workers 4 \
        --model openai/clip-vit-base-patch32 \
        --output ./coreset_output \
        --seed 42 \
        --push-to-hub your-org/docmatix-coreset \
        --translate-model Qwen/Qwen3-4B \
        --tp-size 2 \
        --max-num-seqs 256

Architecture
────────────
No dispatcher. Each worker owns one shard and runs the full pipeline:

    HuggingFace Dataset
    ├── shard 0 → Worker 0 (4 CPU + 0.5 GPU): load → embed → greedy → local coreset
    ├── shard 1 → Worker 1 (4 CPU + 0.5 GPU): load → embed → greedy → local coreset
    ├── shard 2 → Worker 2 (4 CPU + 0.5 GPU): load → embed → greedy → local coreset
    └── shard 3 → Worker 3 (4 CPU + 0.5 GPU): load → embed → greedy → local coreset
                                                              ↓
                                                    driver collects all local coresets
                                                              ↓
                                                    global greedy k-center
                                                              ↓
                                                    final coreset saved to disk
                                                              ↓
                                              (optional) evaluate embedding diversity
                                                              ↓
                                              (optional) translate texts → Cantonese
                                                              ↓
                                               push images + translated texts to Hub

Stopping logic
──────────────
    samples_per_worker = total_samples // num_workers
    n_pages            = samples_per_worker // page_size

    Each worker covers exactly samples_per_worker stream positions
    from its own shard. page_size × n_pages = samples_per_worker.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import os


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Distributed greedy coreset — shard-per-worker edition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    data = p.add_argument_group("Dataset (HuggingFace)")
    data.add_argument("--hf-dataset",   type=str, required=True)
    data.add_argument("--hf-split",     type=str, default="train")
    data.add_argument("--hf-image-col", type=str, default="images",
                      help="Dataset column containing a list of images per example.")
    data.add_argument("--hf-text-col",  type=str, default="texts",
                      help="Dataset column containing conversation turns per example.")
    data.add_argument("--hf-name",      type=str, default=None,
                      help="Optional HF dataset config name.")
    data.add_argument("--hf-token",     type=str, default=None,
                      help="HF access token for gated datasets.")
    data.add_argument(
        "--total-samples", type=int, default=100_000,
        help=(
            "Total stream positions covered across ALL workers. "
            "Each worker covers total_samples // num_workers. "
            "Must be divisible by (num_workers × page_size)."
        ),
    )
    data.add_argument(
        "--page-size", type=int, default=2_000,
        help="Stream positions advanced per page refill per worker.",
    )
    data.add_argument(
        "--sample-size", type=int, default=256,
        help="Examples kept per page (stride = page_size // sample_size).",
    )

    # ── Coreset ───────────────────────────────────────────────────────────────
    core = p.add_argument_group("Coreset")
    core.add_argument("--final-coreset-size", type=int, default=10_000)
    core.add_argument("--local-coreset-size",  type=int, default=1_000,
                      help="Max coreset size per worker between pages.")

    # ── Embedding ─────────────────────────────────────────────────────────────
    emb = p.add_argument_group("Embedding")
    emb.add_argument("--embed-batch-size", type=int, default=128)
    emb.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")

    # ── Compute ───────────────────────────────────────────────────────────────
    comp = p.add_argument_group("Compute")
    comp.add_argument("--workers",         type=int, default=4,
                      help="Number of workers = number of dataset shards.")
    comp.add_argument("--cpus-per-worker", type=int, default=4,
                      help="CPUs per worker (used for data loading).")
    comp.add_argument("--gpus-per-worker", type=float, default=0.5,
                      help="GPU fraction per worker.")

    # ── Output ────────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--output", type=str, default="coreset_output")
    out.add_argument("--seed",   type=int, default=42)

    # ── Hub push + translation ────────────────────────────────────────────────
    hub = p.add_argument_group("Hub push & Cantonese translation")
    hub.add_argument(
        "--push-to-hub", type=str, default=None,
        help="HuggingFace Hub repo id to push the coreset dataset to, e.g. "
             "'your-org/docmatix-coreset'. Leave empty to skip.",
    )
    hub.add_argument(
        "--translate-model", type=str, default="Qwen/Qwen3-4B",
        help="vLLM model used for Cantonese translation (only used when --push-to-hub is set).",
    )
    hub.add_argument(
        "--tp-size", type=int, default=1,
        help=(
            "Tensor-parallel size — GPUs per model replica. "
            "dp_size is derived automatically as total_gpus // tp_size. "
            "E.g. with 4 GPUs and --tp-size 2 you get TP=2, DP=2."
        ),
    )
    hub.add_argument(
        "--max-num-seqs", type=int, default=256,
        help=(
            "vLLM internal batch size — prompts per forward pass per replica. "
            "Raise for A100/H100 (e.g. 512), lower for T4 if OOM (e.g. 128)."
        ),
    )
    hub.add_argument(
        "--hub-token", type=str, default=None,
        help="HF token with write access. Falls back to HF_TOKEN env var.",
    )
    hub.add_argument(
        "--hub-private", action="store_true",
        help="Create the Hub dataset repo as private.",
    )
    hub.add_argument(
        "--push-batch-size", type=int, default=500,
        help=(
            "Number of coreset rows to download, translate, and upload per iteration. "
            "Each batch becomes one parquet shard on the Hub. "
            "Lower values use less RAM; higher values reduce Hub round-trips."
        ),
    )

    return p.parse_args()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_args(args: argparse.Namespace) -> None:
    errors: List[str] = []

    if args.sample_size > args.page_size:
        errors.append(f"--sample-size ({args.sample_size}) must be <= --page-size ({args.page_size})")

    samples_per_worker = args.total_samples // args.workers
    if samples_per_worker < args.page_size:
        errors.append(
            f"samples_per_worker ({samples_per_worker}) = total_samples // workers "
            f"must be >= page_size ({args.page_size})"
        )

    if args.final_coreset_size > args.total_samples:
        errors.append(
            f"--final-coreset-size ({args.final_coreset_size}) > "
            f"--total-samples ({args.total_samples})"
        )

    # Round total_samples to nearest valid multiple
    unit = args.workers * args.page_size
    if args.total_samples % unit != 0:
        rounded = (args.total_samples // unit) * unit
        print(
            f"[Warning] --total-samples ({args.total_samples:,}) is not a multiple of "
            f"workers×page_size ({unit:,}). Rounding down to {rounded:,}."
        )
        args.total_samples = rounded

    if errors:
        for e in errors:
            print(f"[Error] {e}")
        raise SystemExit(1)


# ── Find GPU nodes ────────────────────────────────────────────────────────────

def find_gpu_nodes() -> List[dict]:
    gpu_nodes = [
        node for node in ray.nodes()
        if node["Alive"] and node["Resources"].get("GPU", 0) > 0
    ]
    if not gpu_nodes:
        raise RuntimeError("No GPU nodes found in the Ray cluster.")
    for node in gpu_nodes:
        print(
            f"  [GPU node] {node['NodeManagerAddress']}  "
            f"GPU={node['Resources'].get('GPU')}  "
            f"CPU={node['Resources'].get('CPU')}"
        )
    return gpu_nodes


# ── Global merge ──────────────────────────────────────────────────────────────

def global_merge(
    local_results: List[tuple],
    final_coreset_size: int,
) -> tuple:
    """
    Collect all local (embeddings, ids) from workers and run one final
    greedy k-center pass to produce the global coreset.
    """
    from coreset.greedy import greedy_kcenter

    all_embs, all_ids = [], []
    for embs, ids in local_results:
        if embs is not None and len(embs) > 0:
            all_embs.append(embs)
            all_ids.append(ids)

    if not all_embs:
        raise RuntimeError("No embeddings collected from any worker.")

    pool_embs = np.concatenate(all_embs, axis=0)
    pool_ids  = np.concatenate(all_ids,  axis=0)

    # Deduplicate by stream id
    _, unique_mask = np.unique(pool_ids, return_index=True)
    pool_embs = pool_embs[unique_mask]
    pool_ids  = pool_ids[unique_mask]

    print(f"[Global merge] Pool after dedup: {len(pool_embs):,} candidates")

    k = min(final_coreset_size, len(pool_embs))
    sel, _ = greedy_kcenter(pool_embs, k)
    return pool_embs[sel], pool_ids[sel]


# ── Embedding evaluation ──────────────────────────────────────────────────────

def evaluate_embeddings(embeddings: np.ndarray, output_dir: Path) -> None:
    """
    Compute pairwise cosine similarity statistics over the final coreset
    embeddings and save a histogram of the distribution.

    Args:
        embeddings:  Array of shape (n, d) — the final coreset embeddings.
        output_dir:  Directory where the histogram PNG will be saved.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt

    print(f"\n[Eval] Computing pairwise cosine similarities for {len(embeddings):,} embeddings ...")
    t0 = time.time()

    sim_matrix = cosine_similarity(embeddings)

    # Exclude diagonal (self-similarity = 1.0)
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    pairwise_sims = sim_matrix[mask]

    print(f"[Eval] Done in {time.time() - t0:.1f}s")
    print(f"  Count            : {len(pairwise_sims):,}")
    print(f"  Mean similarity  : {pairwise_sims.mean():.4f}")
    print(f"  Median similarity: {np.median(pairwise_sims):.4f}")
    print(f"  Std similarity   : {pairwise_sims.std():.4f}")
    print(f"  Min similarity   : {pairwise_sims.min():.4f}")
    print(f"  Max similarity   : {pairwise_sims.max():.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pairwise_sims, bins=50, edgecolor="black")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Pairwise Similarity Distribution of Coreset Embeddings")
    fig.tight_layout()

    hist_path = output_dir / "coreset_similarity_histogram.png"
    fig.savefig(hist_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"[Eval] Histogram saved → {hist_path}")


# ── Hub push ──────────────────────────────────────────────────────────────────

def push_coreset_to_hub(
    final_ids: np.ndarray,
    final_embeddings: np.ndarray,
    args: argparse.Namespace,
    n_gpus: int = 1,
) -> None:
    """
    Stream the coreset to the Hub in fixed-size batches so RAM stays bounded
    regardless of how large ``final_ids`` is.

    tp_size and dp_size are derived from n_gpus and args.tp_size:
        tp_size = args.tp_size
        dp_size = n_gpus // tp_size   (e.g. 4 GPUs, tp=2 → dp=2)

        1. Selects the rows from the source HF dataset (download images + texts).
        2. Translates every ``user`` / ``assistant`` turn to Cantonese with vLLM.
        3. Serialises the batch to a numbered parquet shard and uploads it to Hub.

    The Hub dataset ends up with columns:
        ``images``           List[PIL.Image]  — page images for the document
        ``texts``            List[Dict]        — Cantonese-translated conversations
                                                 {"user": ..., "assistant": ..., "source": ...}
        ``texts_en``         List[Dict]        — original English conversations
                                                 {"user": ..., "assistant": ..., "source": ...}
    """
    import tempfile

    import datasets
    from datasets import Dataset, Image as HFImage
    from huggingface_hub import HfApi

    from coreset.translate import CantoneseTranslator

    indices: List[int]  = final_ids.tolist()           # plain Python ints for HF .select()
    n_total             = len(indices)
    batch_sz            = args.push_batch_size
    n_batches           = (n_total + batch_sz - 1) // batch_sz

    # ── Load source dataset (streaming=True for memory efficiency) ────────────
    print(f"\n[Hub] Loading source dataset '{args.hf_dataset}' (split={args.hf_split}) ...")
    load_kwargs: Dict[str, Any] = dict(streaming=True)
    if args.hf_name:
        load_kwargs["name"] = args.hf_name
    if args.hf_token:
        load_kwargs["token"] = args.hf_token

    ds_stream = datasets.load_dataset(
        args.hf_dataset,
        "images",
        split=args.hf_split,
        **load_kwargs,
    )

    target_set = set(indices)
    collected: Dict[int, Any] = {}
    print(f"[Hub] Streaming to collect {len(indices):,} coreset rows ...")
    t0 = time.time()
    for stream_idx, row in enumerate(ds_stream):
        if stream_idx in target_set:
            collected[stream_idx] = row
            if len(collected) == len(indices):
                break
    print(f"[Hub] Collected {len(collected):,} rows in {time.time() - t0:.1f}s")
    ordered_rows = [collected[i] for i in indices]

    # Derive TP / DP from cluster GPU count and user-specified tp_size
    tp_size = args.tp_size
    dp_size = max(1, n_gpus // tp_size)

    # ── Load translation model once — reused across all batches ──────────────
    print(
        f"[Hub] Loading translation model '{args.translate_model}'  "
        f"TP={tp_size}  DP={dp_size}  (total GPUs={n_gpus}) ..."
    )
    translator = CantoneseTranslator(
        model=args.translate_model,
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
    )

    # ── Create Hub repo once so all shard uploads land in the same place ──────
    token = args.hub_token or None
    api   = HfApi(token=token)
    api.create_repo(
        repo_id=args.push_to_hub,
        repo_type="dataset",
        private=args.hub_private,
        exist_ok=True,
    )
    print(f"[Hub] Repo ready → https://huggingface.co/datasets/{args.push_to_hub}")

    # ── Batch loop ────────────────────────────────────────────────────────────
    t_push_total = time.time()

    for batch_idx in range(n_batches):
        lo = batch_idx * batch_sz
        hi = min(lo + batch_sz, n_total)

        batch_indices: List[int] = indices[lo:hi]

        print(
            f"\n[Hub] Batch {batch_idx + 1}/{n_batches}  "
            f"rows {lo:,}–{hi - 1:,}  ({hi - lo} examples)"
        )

        # 1. Download this slice of the dataset ───────────────────────────────
        t0 = time.time()
        batch_rows = ordered_rows[lo:hi]
        raw_images: List[Any]        = [r[args.hf_image_col] for r in batch_rows]
        raw_texts:  List[List[Dict]] = [r[args.hf_text_col]  for r in batch_rows]
        print(f"  Downloaded in {time.time() - t0:.1f}s")

        # 2. Translate user / assistant turns to Cantonese ────────────────────
        t0 = time.time()
        texts_yue: List[List[Dict]] = translator.translate_conversations(
            all_conversations=raw_texts,
            max_num_seqs=args.max_num_seqs,
        )
        print(f"  Translated in {time.time() - t0:.1f}s")

        # 3. Build Dataset for this batch ─────────────────────────────────────
        records = {
            args.hf_image_col:        raw_images,   # List[List[PIL.Image]]
            args.hf_text_col:         texts_yue,    # List[List[Dict]]  — Cantonese
            f"{args.hf_text_col}_en": raw_texts,    # List[List[Dict]]  — English
        }

        batch_hf = Dataset.from_dict(records)

        # Cast image column so Hub viewer renders thumbnails
        try:
            batch_hf = batch_hf.cast_column(
                args.hf_image_col,
                datasets.Sequence(HFImage()),
            )
        except Exception as cast_err:
            print(f"  [Warn] Image cast failed ({cast_err!r}); uploading as-is.")

        # 4. Serialise to a temp parquet file and upload as a named shard ─────
        t0 = time.time()
        shard_name = (
            f"data/train-{batch_idx:05d}-of-{n_batches:05d}.parquet"
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
            batch_hf.to_parquet(tmp.name)
            api.upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=shard_name,
                repo_id=args.push_to_hub,
                repo_type="dataset",
                token=token,
            )
        print(f"  Uploaded shard '{shard_name}' in {time.time() - t0:.1f}s")

    # Release translator actors + placement group now that all batches are done
    translator.shutdown()

    print(
        f"\n[Hub] All {n_batches} shards uploaded in "
        f"{time.time() - t_push_total:.1f}s total  →  "
        f"https://huggingface.co/datasets/{args.push_to_hub}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    validate_args(args)

    samples_per_worker = args.total_samples // args.workers
    n_pages            = samples_per_worker // args.page_size
    stride             = args.page_size // args.sample_size

    np.random.seed(args.seed)

    print("=" * 64)
    print("  Distributed Greedy Coreset  [shard-per-worker edition]")
    print("=" * 64)
    print(f"  HF dataset        : {args.hf_dataset}  (split={args.hf_split})")
    print(f"  Image column      : {args.hf_image_col}")
    print(f"  Text column       : {args.hf_text_col}")
    print(f"  Workers           : {args.workers}  (one shard each)")
    print(f"  Total samples     : {args.total_samples:,}  (all workers combined)")
    print(f"  Samples/worker    : {samples_per_worker:,}")
    print(f"  Page size         : {args.page_size:,}  (stream positions per refill)")
    print(f"  Sample size       : {args.sample_size:,}  (kept per page)")
    print(f"  Stride            : {stride}x")
    print(f"  Pages/worker      : {n_pages}")
    print(f"  Local coreset     : {args.local_coreset_size:,}  (per worker)")
    print(f"  Final coreset     : {args.final_coreset_size:,}")
    print(f"  Embed batch       : {args.embed_batch_size}")
    print(f"  Model             : {args.model}")
    print(f"  CPUs/worker       : {args.cpus_per_worker}")
    print(f"  GPUs/worker       : {args.gpus_per_worker}")
    print(f"  Output            : {args.output}")
    print(f"  Seed              : {args.seed}")
    if args.push_to_hub:
        print(f"  Push to Hub       : {args.push_to_hub}")
        print(f"  Translate model   : {args.translate_model}")
        print(f"  TP size           : {args.tp_size}  (GPUs per replica)")
        print(f"  max_num_seqs      : {args.max_num_seqs}  (vLLM internal batch)")
        print(f"  Push batch size   : {args.push_batch_size:,}  (rows per shard)")
    print("=" * 64)

    # ── Ray init ──────────────────────────────────────────────────────────────
    t_total_start = time.time()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    ray.init(
        address="auto",
        runtime_env={
            "env_vars": {
                "HF_TOKEN": hf_token or "",
            }
        } if hf_token else {},
    )
    print(f"[Ray] Cluster resources: {ray.cluster_resources()}")

    # ── GPU nodes ─────────────────────────────────────────────────────────────
    print(f"\n[Info] Scanning cluster for GPU nodes ...")
    gpu_nodes = find_gpu_nodes()
    total_gpu_capacity = sum(n["Resources"].get("GPU", 0) for n in gpu_nodes)
    max_workers = int(total_gpu_capacity / args.gpus_per_worker)
    if args.workers > max_workers:
        print(
            f"[Warning] --workers ({args.workers}) > cluster capacity "
            f"({total_gpu_capacity} GPUs / {args.gpus_per_worker} = {max_workers} max). "
            f"Clamping to {max_workers}."
        )
        args.workers = max_workers

    # ── HF kwargs ─────────────────────────────────────────────────────────────
    hf_kwargs = {}
    if args.hf_name:
        hf_kwargs["name"] = args.hf_name
    if args.hf_token:
        hf_kwargs["token"] = args.hf_token

    # ── Spawn workers ─────────────────────────────────────────────────────────
    print(f"\n[1/2] Spawning {args.workers} workers ...")
    t0 = time.time()  
    from coreset.CoresetWorker import CoresetWorker
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    workers = []
    for i in range(args.workers):
        node = gpu_nodes[i % len(gpu_nodes)]
        worker = CoresetWorker.options(
            num_gpus=args.gpus_per_worker,
            num_cpus=args.cpus_per_worker,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"],
                soft=False,
            ),
        ).remote(
            worker_id=i,
            num_workers=args.workers,
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            image_col=args.hf_image_col,
            page_size=args.page_size,
            sample_size=args.sample_size,
            total_samples=args.total_samples,
            coreset_size=args.local_coreset_size,
            model_name=args.model,
            embed_batch_size=args.embed_batch_size,
            seed=args.seed,
            **hf_kwargs,
        )
        workers.append(worker)
        print(f"  Worker {i} → shard {i}/{args.workers} on {node['NodeManagerAddress']}")
    
    print(f"[Timing] Worker spawn: {time.time() - t0:.1f}s")

    # ── Run all workers in parallel ───────────────────────────────────────────
    print(f"\n[2/2] Running all workers in parallel ...")
    t_start = time.time()

    # Each worker.run() returns (embeddings, ids) when done
    futures = [w.run.remote() for w in workers]
    local_results = ray.get(futures)   # blocks until ALL workers finish

    t_elapsed = time.time() - t_start
    print(f"\n[Done] All workers finished in {t_elapsed:.1f}s")

    # Print per-worker stats
    for w in workers:
        try:
            s = ray.get(w.stats.remote())
            print(
                f"  Worker {s['worker_id']}: "
                f"pages={s['pages_processed']}  "
                f"coreset_size={s['coreset_size']}  "
                f"skipped={s['images_skipped']}"
            )
        except Exception as exc:
            print(f"  Worker stats unavailable: {exc!r}")

    # ── Global merge ──────────────────────────────────────────────────────────
    print(f"\n[Global] Merging {args.workers} local coresets → {args.final_coreset_size:,} ...")
    t_merge = time.time()
    final_embeddings, final_ids = global_merge(local_results, args.final_coreset_size)
    print(
        f"[Global] Done in {time.time() - t_merge:.1f}s. "
        f"Final coreset: {len(final_ids):,} points."
    )

    t0 = time.time()
    sort_order = np.argsort(final_ids)
    final_ids = final_ids[sort_order]
    final_embeddings = final_embeddings[sort_order]
    print(f"[Sort] Sorted {len(final_ids):,} coreset indices in {time.time() - t0:.3f}s")

    # ── Kill worker actors to free their GPU allocations ──────────────────────
    # ray.shutdown() would also kill the Ray cluster that vLLM needs for tensor
    # parallelism. Instead we kill only the CoresetWorker actors so their GPU
    # memory is released, while the cluster itself stays up.
    print("\n[Ray] Killing CoresetWorker actors to release GPU memory ...")
    for w in workers:
        ray.kill(w)
    print("[Ray] Workers killed.")

    # ── Save ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "coreset_indices.npy",    final_ids)
    np.save(out_dir / "coreset_embeddings.npy", final_embeddings)
    print(f"[Timing] Save to disk: {time.time() - t0:.3f}s")

    print(f"\n[Saved]")
    print(f"  {out_dir / 'coreset_indices.npy'}    — shape {final_ids.shape}")
    print(f"  {out_dir / 'coreset_embeddings.npy'} — shape {final_embeddings.shape}")

    # ── Evaluate coreset diversity ────────────────────────────────────────────
    t0 = time.time()
    evaluate_embeddings(final_embeddings, out_dir)
    print(f"[Timing] Embedding evaluation: {time.time() - t0:.1f}s") 

    # ── Optional: push to Hub with Cantonese translation ──────────────────────
    # vLLM spawns its own Ray actors for tensor parallelism, so Ray must still
    # be running at this point. We shut it down only after translation is done.
    if args.push_to_hub:
        t0 = time.time() 
        push_coreset_to_hub(final_ids, final_embeddings, args, n_gpus=int(total_gpu_capacity))
        print(f"[Timing] Hub push + translation: {time.time() - t0:.1f}s")

    ray.shutdown()

    print(f"\nTotal wall time: {time.time() - t_total_start:.1f}s")


if __name__ == "__main__":
    main()