"""
train.py — Distributed Greedy Coreset Selection (HuggingFace streaming edition)

Usage example:
    python train.py \
        --hf-dataset laion/laion2B-en \
        --hf-split train \
        --hf-image-col image \
        --total-samples 100000 \
        --page-size 10000 \
        --sample-size 1000 \
        --final-coreset-size 10000 \
        --local-coreset-size 1000 \
        --batch-size 256 \
        --embed-batch-size 128 \
        --workers 4 \
        --model openai/clip-vit-base-patch32 \
        --output ./coreset_output \
        --seed 42

Stopping logic
──────────────
    page_size × n_pages = total_samples   (guaranteed)

    e.g. page_size=10,000 × 10 pages = 100,000 total_samples

    sample_size  → how many kept per page  (shuffle quality)
    batch_size   → how workers receive chunks (throughput)
    Neither affects how much of the stream is covered.

Architecture recap
──────────────────
                      ┌─────────────────────────────────────┐
                      │  WorkDispatcher (Ray actor)          │
                      │  • Owns ONE HF streaming iterator    │
   HuggingFace Hub ──►│  • Streams past page_size per refill │
   (4 TB, remote)     │  • Keeps sample_size per page        │
                      │  • Stops after n_pages = total/page  │
                      └───────────────┬─────────────────────┘
                                      │  (images, ids)
                    ┌─────────────────┼────────────────────┐
                    ▼                 ▼                     ▼
             CoresetWorker 0  CoresetWorker 1  … CoresetWorker N
             (0.5 GPU each)   (0.5 GPU each)    (0.5 GPU each)
             CLIP embed        CLIP embed         CLIP embed
             greedy k-center   greedy k-center    greedy k-center
             local coreset     local coreset      local coreset
                    │                 │                     │
                    └─────────────────┴─────────────────────┘
                                      │  global merge
                                      ▼
                           final coreset (indices + embeddings)
"""

import argparse
import math
import time
from pathlib import Path
from typing import List

import numpy as np
import ray


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Distributed greedy coreset selection over a HuggingFace streaming dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    data = p.add_argument_group("Dataset (HuggingFace)")
    data.add_argument(
        "--hf-dataset", type=str, required=True,
        help="HuggingFace dataset repo id, e.g. 'laion/laion2B-en'.",
    )
    data.add_argument(
        "--hf-split", type=str, default="train",
        help="Dataset split to stream from.",
    )
    data.add_argument(
        "--hf-image-col", type=str, default="image",
        help="Column name that contains images (PIL, bytes-dict, or URL string).",
    )
    data.add_argument(
        "--hf-name", type=str, default=None,
        help="Optional dataset config name (the 'name' kwarg for load_dataset).",
    )
    data.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace access token for gated datasets (or set HF_TOKEN env var).",
    )
    data.add_argument(
        "--total-samples", type=int, default=100_000,
        help=(
            "Total stream positions to cover. Must be a multiple of --page-size. "
            "n_pages = total_samples // page_size."
        ),
    )
    data.add_argument(
        "--page-size", type=int, default=10_000,
        help=(
            "Stream positions advanced per refill. "
            "total_samples // page_size = number of pages. "
            "Larger = better dataset coverage per page."
        ),
    )
    data.add_argument(
        "--sample-size", type=int, default=1_000,
        help=(
            "Examples kept per page (stride = page_size // sample_size). "
            "Must be <= page_size. Controls shuffle quality, not stream coverage."
        ),
    )

    # ── Coreset ───────────────────────────────────────────────────────────────
    core = p.add_argument_group("Coreset")
    core.add_argument(
        "--final-coreset-size", type=int, default=10_000,
        help="Size K of the final output coreset (after global merge).",
    )
    core.add_argument(
        "--local-coreset-size", type=int, default=1_000,
        help="Max coreset size maintained per worker between batches.",
    )
    core.add_argument(
        "--batch-size", type=int, default=256,
        help=(
            "Images per dispatcher batch handed to each worker. "
            "Controls throughput only — does not affect stream coverage."
        ),
    )

    # ── Embedding ─────────────────────────────────────────────────────────────
    emb = p.add_argument_group("Embedding")
    emb.add_argument(
        "--embed-batch-size", type=int, default=128,
        help="Mini-batch size for CLIP inference inside each worker.",
    )
    emb.add_argument(
        "--model", type=str, default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name.",
    )

    # ── Compute ───────────────────────────────────────────────────────────────
    comp = p.add_argument_group("Compute")
    comp.add_argument(
        "--workers", type=int, default=2,
        help="Total number of CoresetWorker actors (max 2× physical GPUs).",
    )
    comp.add_argument(
        "--cpus-per-worker", type=int, default=1,
        help="CPUs to reserve per CoresetWorker actor.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--output", type=str, default="coreset_output")
    out.add_argument("--seed", type=int, default=42)
    out.add_argument(
        "--progress-every", type=int, default=20,
        help="Print a progress line every N completed batches.",
    )

    return p.parse_args()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_args(args: argparse.Namespace) -> None:
    errors: List[str] = []

    if args.total_samples < args.page_size:
        errors.append(
            f"--total-samples ({args.total_samples:,}) must be >= "
            f"--page-size ({args.page_size:,})."
        )
    if args.total_samples % args.page_size != 0:
        # round down and warn rather than hard error
        rounded = (args.total_samples // args.page_size) * args.page_size
        print(
            f"[Warning] --total-samples ({args.total_samples:,}) is not a multiple of "
            f"--page-size ({args.page_size:,}). Rounding down to {rounded:,}."
        )
        args.total_samples = rounded

    if args.sample_size > args.page_size:
        errors.append(
            f"--sample-size ({args.sample_size:,}) must be <= "
            f"--page-size ({args.page_size:,})."
        )
    if args.final_coreset_size > args.total_samples:
        errors.append(
            f"--final-coreset-size ({args.final_coreset_size:,}) > "
            f"--total-samples ({args.total_samples:,})."
        )
    if args.local_coreset_size < args.embed_batch_size:
        errors.append("--local-coreset-size should be >= --embed-batch-size.")
    if args.batch_size > args.sample_size:
        print(
            f"[Warning] --batch-size ({args.batch_size}) > --sample-size "
            f"({args.sample_size}). Workers may stall waiting for page refills."
        )

    if errors:
        for e in errors:
            print(f"[Error] {e}")
        raise SystemExit(1)


# ── Find GPU nodes in cluster ─────────────────────────────────────────────────

def find_gpu_nodes() -> List[dict]:
    """Return list of alive nodes that have at least one GPU."""
    gpu_nodes = [
        node for node in ray.nodes()
        if node["Alive"] and node["Resources"].get("GPU", 0) > 0
    ]
    if not gpu_nodes:
        raise RuntimeError(
            "No GPU nodes found in the Ray cluster! "
            "Make sure GPU nodes joined with --num-gpus=N."
        )
    for node in gpu_nodes:
        print(
            f"  [GPU node] {node['NodeManagerAddress']}  "
            f"GPU={node['Resources'].get('GPU')}  "
            f"CPU={node['Resources'].get('CPU')}"
        )
    return gpu_nodes


# ── Work-stealing driver loop ─────────────────────────────────────────────────

def run_work_stealing(workers, dispatcher, progress_every):
    pending_work  = {}   # { work_ref: worker }
    pending_fetch = {}   # { fetch_ref: worker }
    free_workers  = list(workers)   # workers waiting for a batch
    completed = 0
    t0 = time.time()

    # Pre-fetch one batch per worker immediately
    for w in free_workers:
        ref = dispatcher.get_batch.remote()
        pending_fetch[ref] = w

    free_workers = []  # all workers now have a fetch in flight

    while pending_work or pending_fetch:
        all_refs = list(pending_work.keys()) + list(pending_fetch.keys())
        done_list, _ = ray.wait(all_refs, num_returns=1, timeout=None)
        done_ref = done_list[0]

        if done_ref in pending_fetch:
            worker = pending_fetch.pop(done_ref)
            result = ray.get(done_ref)

            if result is None:
                continue  # exhausted, retire this worker

            # Fire the work immediately
            work_ref = worker.process_batch.remote(result["images"], result["ids"])
            pending_work[work_ref] = worker

            # Fire the NEXT fetch immediately too — don't wait for work to finish
            next_ref = dispatcher.get_batch.remote()
            pending_fetch[next_ref] = worker   # same worker, next batch

        else:
            worker = pending_work.pop(done_ref)
            try:
                ray.get(done_ref)
            except ray.exceptions.ActorDiedError as exc:
                print(f"[Worker] died: {exc!r}")
                continue
            except Exception as exc:
                print(f"[Worker] error: {exc!r}")

            completed += 1
            if completed % progress_every == 0:
                prog = ray.get(dispatcher.progress.remote())
                elapsed = time.time() - t0
                issued  = prog["images_issued"]
                total   = prog["total_samples"]
                rate    = issued / elapsed if elapsed > 0 else 0
                eta     = (prog["remaining"] / rate) if rate > 0 else float("inf")
                print(
                    f"[Progress] {prog['pct']:.1f}%  |  "
                    f"{issued:,}/{total:,} images  |  "
                    f"{elapsed:.0f}s elapsed  |  ETA {eta:.0f}s"
                )
                
# def run_work_stealing(workers: list, dispatcher, progress_every: int) -> None:
#     pending_work  = {}   # { process_batch_future: worker }
#     pending_fetch = {}   # { get_batch_future:     worker }
#     completed = 0
#     t0 = time.time()

#     # Seed: fire one non-blocking dispatcher fetch per worker
#     for w in workers:
#         ref = dispatcher.get_batch.remote()
#         pending_fetch[ref] = w

#     while pending_work or pending_fetch:
#         all_refs = list(pending_work.keys()) + list(pending_fetch.keys())
#         done_list, _ = ray.wait(all_refs, num_returns=1, timeout=None)
#         done_ref = done_list[0]

#         # ── A dispatcher fetch completed ──────────────────────────────────────
#         if done_ref in pending_fetch:
#             worker = pending_fetch.pop(done_ref)
#             try:
#                 result = ray.get(done_ref)
#             except Exception as exc:
#                 print(f"[Dispatcher] fetch error: {exc!r}")
#                 continue

#             if result is None:
#                 # Dispatcher exhausted — this worker is done, don't requeue
#                 continue

#             # Immediately pipeline the NEXT fetch for this worker
#             next_fetch = dispatcher.get_batch.remote()
#             pending_fetch[next_fetch] = worker

#             # Fire the compute work (non-blocking — just enqueues on Ray)
#             work_ref = worker.process_batch.remote(result["images"], result["ids"])
#             pending_work[work_ref] = worker

#         # ── A worker finished its batch ───────────────────────────────────────
#         else:
#             worker = pending_work.pop(done_ref)
#             try:
#                 ray.get(done_ref)
#             except ray.exceptions.ActorDiedError as exc:
#                 print(f"[Worker] Actor died permanently, dropping: {exc!r}")
#                 continue
#             except Exception as exc:
#                 print(f"[Worker] process_batch raised: {exc!r} — continuing")

#             completed += 1
#             if completed % progress_every == 0:
#                 prog = ray.get(dispatcher.progress.remote())
#                 elapsed = time.time() - t0
#                 issued  = prog["images_issued"]
#                 total   = prog["total_samples"]
#                 rate    = issued / elapsed if elapsed > 0 else 0
#                 eta     = (prog["remaining"] / rate) if rate > 0 else float("inf")
#                 print(
#                     f"[Progress] {prog['pct']:.1f}%  |  "
#                     f"{issued:,}/{total:,} images  |  "
#                     f"{elapsed:.0f}s elapsed  |  ETA {eta:.0f}s"
#                 )


# ── Global merge ──────────────────────────────────────────────────────────────

def global_merge(workers: list, final_coreset_size: int) -> tuple:
    from coreset.greedy import greedy_kcenter

    all_embs, all_ids = [], []
    for w in workers:
        try:
            embs, ids = ray.get(w.get_coreset.remote())
            if embs is not None and len(embs) > 0:
                all_embs.append(embs)
                all_ids.append(ids)
        except Exception as exc:
            print(f"[Global merge] Could not retrieve coreset from worker: {exc!r}")

    if not all_embs:
        raise RuntimeError("No embeddings collected — nothing to merge.")

    pool_embs = np.concatenate(all_embs, axis=0)
    pool_ids  = np.concatenate(all_ids,  axis=0)

    # Deduplicate by stream ID
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

    n_pages = args.total_samples // args.page_size
    stride  = args.page_size // args.sample_size

    np.random.seed(args.seed)

    print("=" * 64)
    print("  Distributed Greedy Coreset Selection  [HF Streaming]")
    print("=" * 64)
    print(f"  HF dataset      : {args.hf_dataset}  (split={args.hf_split})")
    print(f"  Image column    : {args.hf_image_col}")
    print(f"  Total samples   : {args.total_samples:,}  (stream positions covered)")
    print(f"  Page size       : {args.page_size:,}  (positions per refill)")
    print(f"  Sample size     : {args.sample_size:,}  (kept per page)")
    print(f"  Stride          : {stride}x  (1 in {stride} kept)")
    print(f"  n_pages         : {n_pages}  (page_size × n_pages = total_samples)")
    print(f"  Batch size      : {args.batch_size:,}  (images per worker round)")
    print(f"  Final coreset   : {args.final_coreset_size:,}")
    print(f"  Local coreset   : {args.local_coreset_size:,}  (per worker)")
    print(f"  Embed batch     : {args.embed_batch_size}  (images per CLIP call)")
    print(f"  Workers         : {args.workers}  (0.5 GPU each)")
    print(f"  Model           : {args.model}")
    print(f"  Output          : {args.output}")
    print(f"  Seed            : {args.seed}")
    print("=" * 64)

    # ── Ray init ──────────────────────────────────────────────────────────────
    ray.init(address="auto")
    t_start = time.time()
    print(f"[Ray] Cluster resources: {ray.cluster_resources()}")

    # ── Find real GPU nodes and validate worker count ─────────────────────────
    print(f"\n[Info] Scanning cluster for GPU nodes ...")
    gpu_nodes = find_gpu_nodes()
    total_gpu_capacity = sum(n["Resources"].get("GPU", 0) for n in gpu_nodes)
    max_workers = int(total_gpu_capacity * 2)
    if args.workers > max_workers:
        print(
            f"[Warning] --workers ({args.workers}) > cluster GPU capacity "
            f"({total_gpu_capacity} GPUs × 2 = {max_workers} max). "
            f"Clamping to {max_workers}."
        )
        args.workers = max_workers

    # ── HF kwargs ─────────────────────────────────────────────────────────────
    hf_kwargs = {}
    if args.hf_name:
        hf_kwargs["name"] = args.hf_name
    if args.hf_token:
        hf_kwargs["token"] = args.hf_token

    # ── Step 1: Dispatcher ────────────────────────────────────────────────────
    print(f"\n[1/3] Launching dispatcher ...")
    from coreset.dispatcher import WorkDispatcher

    dispatcher = WorkDispatcher.options(num_cpus=3).remote(
        dataset_name=args.hf_dataset,
        split=args.hf_split,
        image_col=args.hf_image_col,
        page_size=args.page_size,
        sample_size=args.sample_size,
        total_samples=args.total_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        **hf_kwargs,
    )
    total_batches = math.ceil(args.total_samples / args.batch_size)
    print(f"      {n_pages} pages × {args.page_size:,} positions = {args.total_samples:,} total.")
    print(f"      ~{total_batches} batches of {args.batch_size} images each.")

    # ── Step 2: Spawn workers pinned to real GPU nodes ────────────────────────
    print(f"\n[2/3] Spawning {args.workers} CoresetWorkers (0.5 GPU each) ...")
    from coreset.CoresetWorker import CoresetWorker
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    workers = []
    for i in range(args.workers):
        node = gpu_nodes[i % len(gpu_nodes)]
        worker = CoresetWorker.options(
            num_gpus=0.5,
            num_cpus=args.cpus_per_worker,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"],
                soft=False,
            ),
        ).remote(
            worker_id=i,
            coreset_size=args.local_coreset_size,
            model_name=args.model,
            embed_batch_size=args.embed_batch_size,
        )
        workers.append(worker)
        print(f"  Worker {i} → {node['NodeManagerAddress']}")

    # ── Step 3: Work-stealing loop ────────────────────────────────────────────
    print(f"\n[3/3] Processing batches (work-stealing) ...\n")
    
    run_work_stealing(workers, dispatcher, args.progress_every)
    t_elapsed = time.time() - t_start
    print(f"\n[Done] All batches processed in {t_elapsed:.1f}s")

    ray.kill(dispatcher)
    dispatcher = None
    print("[Info] Dispatcher released (3 CPUs freed).")

    for w in workers:
        try:
            s = ray.get(w.stats.remote())
            print(
                f"  Worker {s['worker_id']}: "
                f"{s['batches_processed']} batches, "
                f"coreset_size={s['coreset_size']}, "
                f"skipped={s['images_skipped']} images"
            )
        except Exception as exc:
            print(f"  Worker stats unavailable: {exc!r}")

    # ── Global merge ──────────────────────────────────────────────────────────
    print(f"\n[Global] Merging → selecting {args.final_coreset_size:,} ...")
    t_merge = time.time()
    final_embeddings, final_ids = global_merge(workers, args.final_coreset_size)
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