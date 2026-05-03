"""
train.py — Distributed Greedy Coreset Selection (shard-per-worker edition)

Usage example:
    python train.py \
        --hf-dataset HuggingFaceM4/Docmatix \
        --hf-split train \
        --hf-image-col images \
        --total-samples 100000 \
        --page-size 2000 \
        --sample-size 256 \
        --final-coreset-size 10000 \
        --local-coreset-size 1000 \
        --embed-batch-size 128 \
        --workers 4 \
        --model openai/clip-vit-base-patch32 \
        --output ./coreset_output \
        --seed 42

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
from typing import List

import numpy as np
import ray


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
    data.add_argument("--hf-image-col", type=str, default="image")
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
    print("=" * 64)

    # ── Ray init ──────────────────────────────────────────────────────────────
    ray.init(address="auto")
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

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "coreset_indices.npy",    final_ids)
    np.save(out_dir / "coreset_embeddings.npy", final_embeddings)
    

    print(f"\n[Saved]")
    print(f"  {out_dir / 'coreset_indices.npy'}    — shape {final_ids.shape}")
    print(f"  {out_dir / 'coreset_embeddings.npy'} — shape {final_embeddings.shape}")
    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")
    ray.shutdown()

    


if __name__ == "__main__":
    main()
