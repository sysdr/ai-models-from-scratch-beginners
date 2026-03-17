"""
ScratchAI-Beginner | Lesson 01: NumPy Playground
CLI training / demonstration script.
Usage: python train.py --epochs 50 --lr 0.01 --demo
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
from model import (
    dot_product, cosine_similarity, matrix_multiply,
    broadcasting_demo, vectorization_benchmark,
)

def run_operation_suite(epochs: int, lr: float, demo: bool) -> None:
    """
    Demonstrate NumPy array operations with epoch-style logging.
    In Lesson 01 there is no training loop per se — we use 'epochs'
    to show how operation results evolve as we mutate a weight matrix
    via a simple gradient step (W = W - lr * grad).
    """
    rng = np.random.default_rng(0)
    print(f"ScratchAI Lesson 01 | {epochs=} {lr=}")
    print("─" * 60)

    # Synthetic task: learn a target matrix T via mean-squared gradient
    m, n, k = 8, 16, 4
    W = rng.standard_normal((n, k)) * 0.1   # weight matrix
    T = rng.standard_normal((n, k))           # target
    X = rng.standard_normal((m, n))           # input batch

    best_loss = float("inf")
    best_W: np.ndarray | None = None

    for epoch in range(1, epochs + 1):
        # Forward: project input through W
        out = X @ W                        # (m, k)
        target_out = X @ T                 # (m, k)

        # MSE loss
        diff = out - target_out            # (m, k)
        loss = float(np.mean(diff ** 2))

        # Gradient of MSE w.r.t. W: (2/m) * X.T @ diff
        grad = (2.0 / m) * (X.T @ diff)   # (n, k)
        grad_norm = float(np.linalg.norm(grad))

        # Weight update
        W = W - lr * grad

        # Best checkpoint
        if loss < best_loss:
            best_loss = loss
            best_W = W.copy()

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(
                f"  epoch {epoch:>4d}/{epochs}  "
                f"loss={loss:.6f}  "
                f"grad_norm={grad_norm:.4f}  "
                f"{loss < best_loss + 1e-10}"
            )

    print("─" * 60)
    print(f"Final loss:    {loss:.6f}")
    print(f"Best loss:     {best_loss:.6f}")

    if best_W is not None:
        np.save("best_weights.npy", best_W)
        print(f"Saved: best_weights.npy  shape={best_W.shape} dtype={best_W.dtype}")

    if demo:
        print()
        print("── Vectorization Benchmark ─────────────────────────────")
        bench = vectorization_benchmark(200_000)
        print(
            f"  Loop:        {bench['loop_time_ms']:.1f} ms"
            f"  Vectorized:  {bench['vec_time_ms']:.2f} ms"
            f"  Speedup:     {bench['speedup']:.0f}×"
            f"  Identical:   {bench['identical']}"
        )

        print()
        print("── Cosine Similarity ───────────────────────────────────")
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        cs = cosine_similarity(a, b)
        print(
            f"  a={list(a)}  b={[round(x, 4) for x in b.tolist()]}"
            f"  similarity={cs['similarity']:.4f}  (expected ~0.7071)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ScratchAI Lesson 01 — NumPy operation demo"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--demo", action="store_true",
                        help="Run additional demos after training")
    args = parser.parse_args()

    match (args.epochs > 0, args.lr > 0):
        case (True, True):
            run_operation_suite(args.epochs, args.lr, args.demo)
        case (False, _):
            print("Error: --epochs must be > 0", file=sys.stderr)
            sys.exit(1)
        case (_, False):
            print("Error: --lr must be > 0", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()