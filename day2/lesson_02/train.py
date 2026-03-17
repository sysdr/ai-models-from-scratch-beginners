"""
train.py — Tensor Benchmark & Audit Script
ScratchAI Beginner · Lesson 01: Data as Tensors

Usage:
  python train.py                         # default: all ranks, 3 seeds
  python train.py --epochs 50 --lr 0.01 --demo
  python train.py --rank 4 --seed 7

This script benchmarks slice extraction and axis-stat computation
across tensor ranks and logs a full structural audit per tensor.
"Training" here means the iterative process of loading, validating,
and profiling tensors — the data-pipeline analog of a training loop.
"""

from __future__ import annotations
import argparse
import time
import numpy as np

from model import (
    make_demo_tensor,
    compute_shape_stats,
    compute_axis_stats,
    extract_slice,
    audit_array,
    generate_reshape_candidates,
    validate_reshape,
)

BEST_WEIGHTS_PATH = "best_weights.npy"


def log(rank: int, seed: int, epoch: int, elapsed_ms: float, audit: dict, profile) -> None:
    nan_flag = "⚠ NaN" if audit["nan_count"] > 0 else "ok"
    inf_flag = "⚠ Inf" if audit["inf_count"] > 0 else "ok"
    print(
        f"[epoch={epoch:03d}]  "
        f"rank={rank}  seed={seed}  "
        f"shape={str(profile.shape):<20}  "
        f"elements={profile.n_elements:>8,}  "
        f"mem={profile.nbytes/1024:>7.1f}KB  "
        f"contiguous={str(profile.is_contiguous):<5}  "
        f"nan={nan_flag}  inf={inf_flag}  "
        f"elapsed={elapsed_ms:.2f}ms  "
        f"range=[{audit['value_min']:.3f}, {audit['value_max']:.3f}]"
    )


def run_benchmark(ranks: list[int], seeds: list[int], epochs: int) -> None:
    print("=" * 100)
    print("ScratchAI Lesson 01 — Tensor Benchmark")
    print("=" * 100)

    best_score: float = float("inf")
    best_tensor: np.ndarray | None = None

    for epoch in range(1, epochs + 1):
        for rank in ranks:
            for seed in seeds:
                t0 = time.perf_counter()

                arr = make_demo_tensor(rank=rank, seed=seed + epoch)
                profile = compute_shape_stats(arr)
                audit = audit_array(arr)

                # Vectorized axis stats for all axes
                for ax in range(min(profile.rank, 3)):
                    _ = compute_axis_stats(arr, axis=ax)

                # Slice extraction
                if profile.rank >= 2:
                    sa = (profile.rank - 2, profile.rank - 1)
                    _ = extract_slice(arr, slice_axes=sa)

                # Reshape candidate generation
                _ = generate_reshape_candidates(arr)

                elapsed = (time.perf_counter() - t0) * 1000

                log(rank, seed, epoch, elapsed, audit, profile)

                # Track "best" by smallest value range (most stable data)
                score = audit["value_range"] if np.isfinite(audit["value_range"]) else float("inf")
                if score < best_score:
                    best_score = score
                    best_tensor = arr.copy()

    print("=" * 100)
    if best_tensor is not None:
        np.save(BEST_WEIGHTS_PATH, best_tensor)
        print(f"Best tensor saved → {BEST_WEIGHTS_PATH}  "
              f"(shape={best_tensor.shape}, range={best_score:.4f})")
    print("Benchmark complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ScratchAI L01 Tensor Benchmark")
    parser.add_argument("--epochs", type=int, default=3, help="Number of benchmark epochs")
    parser.add_argument("--rank",   type=int, default=None, help="Single rank to benchmark (0–4)")
    parser.add_argument("--seed",   type=int, default=42,   help="Starting random seed")
    parser.add_argument("--demo",   action="store_true",    help="Demo mode: ranks 2+3 only, 2 seeds")
    parser.add_argument("--lr",     type=float, default=0.01, help="(unused — reserved for future lessons)")
    args = parser.parse_args()

    match (args.demo, args.rank):
        case (True, _):
            ranks = [2, 3]
            seeds = [args.seed, args.seed + 1]
        case (False, int(r)) if 0 <= r <= 4:
            ranks = [r]
            seeds = [args.seed]
        case (False, None):
            ranks = [0, 1, 2, 3, 4]
            seeds = [args.seed, args.seed + 1, args.seed + 2]
        case _:
            print(f"Invalid --rank {args.rank}. Must be 0–4.")
            return

    run_benchmark(ranks=ranks, seeds=seeds, epochs=args.epochs)


if __name__ == "__main__":
    main()
