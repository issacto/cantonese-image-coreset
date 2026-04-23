%%writefile train.py
"""
train.py — Distributed Greedy Coreset Selection (HuggingFace streaming edition)

Usage example:
    python train.py \
        --hf-dataset laion/laion2B-en \
        --hf-split train \
        --hf-image-col image \
        --total-samples 100000 \
        --shuffle-buffer 20000 \
        --final-coreset-size 10000 \
        --local-coreset-size 1000 \
        --batch-size 256 \
        --embed-batch-size 16 \
        --workers 4 \
        --gpus 2 \
                            --model openai/clip-vit-base-patch32 \
        --output ./coreset_output \
        --seed 42

Architecture recap
──────────────────
                      ┌─────────────────────────────────────┐
                      │  WorkDispatcher (Ray actor)          │
                      │  • Owns ONE HF streaming iterator    │
   HuggingFace Hub ──►│  • shuffle buffer (RAM, configurable)│
   (4 TB, remote)     │  • Emits batches as Ray object refs  │
                      └───────────────┬─────────────────────┘
                                      │  (images_ref, ids_ref)
                    ┌─────────────────┼────────────────────┐
                    ▼                 ▼                     ▼
             CorsetWorker 0   CorsetWorker 1  …  CorsetWorker N
             (0.5 GPU each)   (0.5 GPU each)     (0.5 GPU each)
             CLIP embed        CLIP embed          CLIP embed
             greedy k-center   greedy k-center     greedy k-center
             local coreset     local coreset       local coreset
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
from typing import List, Optional

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
        "--total-cpus", type=int, required=True,
        help="Total Cpus",
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
        help="Stop after streaming this many images.  None streams until dataset end.",
    )
    data.add_argument(
        "--shuffle-buffer", type=int, default=10_000,
        help=(
            "HF streaming shuffle reservoir size.  Larger = better global shuffle "
            "at the cost of more RAM on the dispatcher node.  "
            "Rule of thumb: 5–10× batch-size, max ~50 000 for typical machines."
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
            "Images per dispatcher batch.  After embedding, the worker prunes "
            "its buffer back to --local-coreset-size."
        ),
    )

    # ── Embedding ─────────────────────────────────────────────────────────────
    emb = p.add_argument_group("Embedding")
    emb.add_argument(
        "--embed-batch-size", type=int, default=16,
        help="Mini-batch size for CLIP inference inside each worker.",
    )
    emb.add_argument(
        "--model", type=str, default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name.",
    )

    # ── Compute ───────────────────────────────────────────────────────────────
    comp = p.add_argument_group("Compute")
    comp.add_argument("--workers", type=int, default=4)
    comp.add_argument(
        "--gpus", type=int, default=1.0,
        help="Total GPUs declared to Ray (e.g. 2.0 for two physical GPUs).",
    )
    comp.add_argument("--cpus-per-worker", type=int, default=1)

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

    if args.total_samples and args.final_coreset_size > args.total_samples:
        errors.append(
            f"--final-coreset-size ({args.final_coreset_size:,}) > "
            f"--total-samples ({args.total_samples:,})"
        )
    if args.local_coreset_size < args.embed_batch_size:
        errors.append(
            "--local-coreset-size should be >= --embed-batch-size."
        )
    # max_workers = math.floor(args.gpus / args.gpu_per_worker)
    # if args.workers > max_workers:
    #     errors.append(
    #         f"--workers ({args.workers}) exceeds GPU capacity "
    #         f"floor({args.gpus}/{args.gpu_per_worker})={max_workers}."
    #     )
    if args.shuffle_buffer < args.batch_size:
        print(
            f"[Warning] --shuffle-buffer ({args.shuffle_buffer}) < --batch-size "
            f"({args.batch_size}).  Shuffle quality will be poor.  "
            "Consider increasing --shuffle-buffer."
        )

    if errors:
        for e in errors:
            print(f"[Error] {e}")
        raise SystemExit(1)


# ── Work-stealing driver loop ─────────────────────────────────────────────────

def run_work_stealing(workers: list, dispatcher, progress_every: int) -> None:
    """
    Seed every worker with its first batch, then reassign workers as they
    finish.  Workers that find no more work drop out of the pending set.

    The dispatcher pre-fetches the next batch while workers are computing —
    GPU and network I/O overlap.
    """
    pending = {}   # { future_ref: worker_actor }
    completed = 0
    t0 = time.time()

    def _dispatch_to(worker):
        # NOTE: get_batch is called synchronously here which is fine —
        # the dispatcher is a Ray actor so this is a remote call, not a
        # local block.  Use a generous timeout; HF streaming can be slow.
        try:
            result = ray.get(dispatcher.get_batch.remote(), timeout=None)
        except ray.exceptions.GetTimeoutError:
            print("[Dispatcher] timed out — treating as exhausted")
            return False

        if result is None:
            return False

        future = worker.process_batch.remote(result["images"], result["ids"])
        pending[future] = worker
        return True

    # Seed all workers
    for w in workers:
        _dispatch_to(w)

    while pending:
        print("completed ", completed, "pending ",pending)

        # ── Wait for any one worker to finish ─────────────────────────────
        # timeout=None here is intentional: we already have max_concurrency=4
        # on CoresetWorker and the embed call properly yields the event loop,
        # so a worker *will* eventually finish.  A finite timeout would cause
        # the "stuck at 199" symptom when the last batch takes slightly longer
        # (e.g. smaller batch, GPU still warm from previous call).
        done_list, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=None)

        done_ref = done_list[0]
        worker = pending.pop(done_ref)

        # ── Surface worker-side exceptions without killing the loop ───────
        # Previously: ray.get(done_ref) would raise and exit the function,
        # leaving remaining futures in `pending` permanently — the "stuck at
        # 199" case when the last worker's process_batch raised an error.
        try:
            ray.get(done_ref)
        except Exception as exc:
            print(f"[Worker] process_batch raised: {exc!r} — continuing")
            # Worker is still alive; re-dispatch to it so it isn't lost
            _dispatch_to(worker)
            continue

        completed += 1

        if completed % progress_every == 0:
            prog = ray.get(dispatcher.progress.remote())
            elapsed = time.time() - t0
            issued = prog["images_issued"]
            total = prog["total_samples"] or "?"
            pct_str = f"{prog['pct']:.1f}%" if prog["total_samples"] else "?%"
            remaining = prog.get("remaining")
            rate = issued / elapsed if elapsed > 0 else 0
            eta = (remaining / rate) if (remaining and rate > 0) else float("inf")
            print(
                f"[Progress] {pct_str}  |  "
                f"{issued:,}/{total} images  |  "
                f"{elapsed:.0f}s elapsed  |  "
                f"ETA {eta:.0f}s"
            )

        # Re-dispatch this worker; if exhausted it simply won't be added
        # back to `pending` and will drop out of the loop naturally.
        _dispatch_to(worker)
# ── Global merge ──────────────────────────────────────────────────────────────

def global_merge(workers: list, final_coreset_size: int) -> tuple:
    from coreset.greedy import greedy_kcenter

    all_embs, all_ids = [], []
    for w in workers:
        embs, ids = ray.get(w.get_coreset.remote())
        if embs is not None and len(embs) > 0:
            all_embs.append(embs)
            all_ids.append(ids)

    if not all_embs:
        raise RuntimeError("No embeddings collected — nothing to merge.")

    pool_embs = np.concatenate(all_embs, axis=0)
    pool_ids = np.concatenate(all_ids, axis=0)

    # Deduplicate by stream ID (same image may appear in two worker coresets)
    _, unique_mask = np.unique(pool_ids, return_index=True)
    pool_embs = pool_embs[unique_mask]
    pool_ids = pool_ids[unique_mask]

    print(f"[Global merge] Pool after dedup: {len(pool_embs):,} candidates")

    k = min(final_coreset_size, len(pool_embs))
    sel, _ = greedy_kcenter(pool_embs, k)
    return pool_embs[sel], pool_ids[sel]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    validate_args(args)

    np.random.seed(args.seed)

    print("=" * 64)
    print("  Distributed Greedy Coreset Selection  [HF Streaming]")
    print("=" * 64)
    print(f"  HF dataset      : {args.hf_dataset}  (split={args.hf_split})")
    print(f"  Image column    : {args.hf_image_col}")
    print(f"  Total samples   : {args.total_samples:,}")
    print(f"  Shuffle buffer  : {args.shuffle_buffer:,}")
    print(f"  Final coreset   : {args.final_coreset_size:,}")
    print(f"  Local coreset   : {args.local_coreset_size:,}  (per worker)")
    print(f"  Batch size      : {args.batch_size:,}  (images per round)")
    print(f"  Embed batch     : {args.embed_batch_size}  (images per CLIP call)")
    print(f"  Model           : {args.model}")
    print(f"  Output          : {args.output}")
    print(f"  Seed            : {args.seed}")
    print("=" * 64)

    # ── Ray init ──────────────────────────────────────────────────────────────
    ray.init(num_cpus=args.total_cpus, num_gpus=args.gpus)
    print(f"[Ray] Cluster resources: {ray.cluster_resources()}")
    args.total_samples = (args.total_samples // args.batch_size) * args.batch_size
    print(f"[Info] Rounded total_samples to {args.total_samples} (multiple of batch_size)")

    # ── HF kwargs ─────────────────────────────────────────────────────────────
    hf_kwargs = {}
    if args.hf_name:
        hf_kwargs["name"] = args.hf_name
    if args.hf_token:
        hf_kwargs["token"] = args.hf_token

    # EMBED
    MODEL = "openai/clip-vit-base-patch32"
    EMBED_BATCH = 128

    from coreset.ClipWorker import ClipWorker

    embedders = [
        ClipWorker.options(num_gpus=1.0).remote(MODEL, EMBED_BATCH)
        for _ in range(args.gpus)   # 2 embedders → 2 GPUs fully utilised
    ]



    # ── Step 1: Dispatcher (owns the HF stream) ───────────────────────────────
    print(f"\n[1/3] Launching dispatcher ...")
    from coreset.dispatcher import WorkDispatcher

    dispatcher = WorkDispatcher.remote(
        dataset_name=args.hf_dataset,
        split=args.hf_split,
        image_col=args.hf_image_col,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        batch_size=args.batch_size,
        total_samples=args.total_samples,
        **hf_kwargs,
    )
    total_batches = (
        math.ceil(args.total_samples / args.batch_size)
        if args.total_samples
        else "?"
    )
    print(f"      ~{total_batches} batches of {args.batch_size} images each.")

    # ── Step 2: Spawn workers ─────────────────────────────────────────────────
    print(f"\n[2/3] Spawning {args.workers} workers ...")
    from coreset.CoresetWorker import CoresetWorker

    WorkerCls = CoresetWorker.options(
        num_cpus=args.cpus_per_worker
    )
    workers = [
        WorkerCls.remote(
            worker_id=i,
            coreset_size=args.local_coreset_size,
            embedder=embedders[i % len(embedders)],
        )
        for i in range(args.workers)
    ]

    # ── Step 3: Work-stealing loop ────────────────────────────────────────────
    print(f"\n[3/3] Processing batches (work-stealing) ...\n")
    t_start = time.time()
    run_work_stealing(workers, dispatcher, args.progress_every)
    t_elapsed = time.time() - t_start
    print(f"\n[Done] All batches processed in {t_elapsed:.1f}s")

    for w in workers:
        s = ray.get(w.stats.remote())
        print(
            f"  Worker {s['worker_id']}: "
            f"{s['batches_processed']} batches, "
            f"coreset_size={s['coreset_size']}, "
            f"skipped={s['images_skipped']} images"
        )

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
    np.save(out_dir / "coreset_indices.npy", final_ids)
    np.save(out_dir / "coreset_embeddings.npy", final_embeddings)

    print(f"\n[Saved]")
    print(f"  {out_dir / 'coreset_indices.npy'}    — shape {final_ids.shape}")
    print(f"  {out_dir / 'coreset_embeddings.npy'} — shape {final_embeddings.shape}")
    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")
    ray.shutdown()


if __name__ == "__main__":
    main()
