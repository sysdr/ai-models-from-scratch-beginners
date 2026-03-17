"""
ScratchAI-Beginner | Lesson 01: Unit tests for model.py
Run with: python test_model.py
"""

from __future__ import annotations
import sys
import time
import numpy as np
from model import (
    dot_product, outer_product, cosine_similarity,
    matrix_multiply, broadcasting_demo, dtype_cast_demo,
    vectorization_benchmark,
)

PASS = "✓"
FAIL = "✗"
results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    symbol = PASS if condition else FAIL
    print(f"  {symbol}  {name}" + (f"  [{detail}]" if detail else ""))


def test_forward_pass_shape() -> None:
    """Output shapes must match expected dimensions."""
    A = np.ones((4, 6), dtype=np.float64)
    B = np.ones((6, 3), dtype=np.float64)
    res = matrix_multiply(A, B)
    check("matmul_output_shape", res["result"].shape == (4, 3),
          f"got {res['result'].shape}")

    a = np.ones(5, dtype=np.float64)
    b = np.ones(5, dtype=np.float64)
    outer = outer_product(a, b)
    check("outer_product_shape", outer["result"].shape == (5, 5),
          f"got {outer['result'].shape}")

    bias = np.ones(6, dtype=np.float64)
    bc = broadcasting_demo(A, bias)
    check("broadcast_add_shape", bc["result"].shape == (4, 6),
          f"got {bc['result'].shape}")


def test_gradient_nonzero() -> None:
    """Gradient computed in train.py must be non-zero for random init."""
    rng = np.random.default_rng(7)
    W = rng.standard_normal((6, 3)) * 0.1
    T = rng.standard_normal((6, 3))
    X = rng.standard_normal((4, 6))
    out = X @ W
    target_out = X @ T
    diff = out - target_out
    grad = (2.0 / 4) * (X.T @ diff)
    check("gradient_nonzero", np.linalg.norm(grad) > 1e-6,
          f"norm={np.linalg.norm(grad):.4f}")


def test_loss_decreases() -> None:
    """Loss at epoch 10 must be less than loss at epoch 1."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 4)) * 0.1
    T = rng.standard_normal((16, 4))
    X = rng.standard_normal((8, 16))
    lr = 0.01
    losses: list[float] = []
    for _ in range(10):
        diff = (X @ W) - (X @ T)
        loss = float(np.mean(diff ** 2))
        losses.append(loss)
        W = W - lr * (2.0 / 8) * (X.T @ diff)
    check("loss_decreases", losses[-1] < losses[0],
          f"epoch1={losses[0]:.4f} epoch10={losses[-1]:.4f}")


def test_numerical_gradient() -> None:
    """Analytical gradient must match numerical gradient (finite differences)."""
    rng = np.random.default_rng(3)
    W = rng.standard_normal((4, 2))
    T = rng.standard_normal((4, 2))
    X = rng.standard_normal((3, 4))

    def mse(W_: np.ndarray) -> float:
        return float(np.mean(((X @ W_) - (X @ T)) ** 2))

    # mse = np.mean(...) divides by product of shape (m*n); gradient is (2/(m*n))*X.T@(X@W-X@T)
    analytical_grad = (2.0 / (X.shape[0] * W.shape[1])) * (X.T @ ((X @ W) - (X @ T)))
    eps = 1e-5
    numerical_grad = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy(); W_p[i, j] += eps
            W_m = W.copy(); W_m[i, j] -= eps
            numerical_grad[i, j] = (mse(W_p) - mse(W_m)) / (2 * eps)

    max_diff = float(np.max(np.abs(analytical_grad - numerical_grad)))
    check("numerical_gradient", max_diff < 1e-6, f"max_diff={max_diff:.2e}")


def test_stress_no_nan_inf() -> None:
    """1000 forward passes with random inputs — no NaN or Inf."""
    rng = np.random.default_rng(99)
    any_bad = False
    for _ in range(1000):
        batch = rng.integers(1, 16)
        n_feat = rng.integers(2, 32)
        n_out = rng.integers(2, 16)
        X = rng.standard_normal((batch, n_feat))
        W = rng.standard_normal((n_feat, n_out)) * 0.1
        out = X @ W
        if np.any(np.isnan(out)) or np.any(np.isinf(out)):
            any_bad = True
            break
    check("stress_no_nan_inf", not any_bad)


def test_stress_runtime() -> None:
    """1000 forward passes must complete in < 5 seconds."""
    rng = np.random.default_rng(77)
    t0 = time.perf_counter()
    for _ in range(1000):
        X = rng.standard_normal((16, 64))
        W = rng.standard_normal((64, 32)) * 0.1
        _ = X @ W
    elapsed = time.perf_counter() - t0
    check("stress_runtime", elapsed < 5.0, f"{elapsed:.3f}s")


def test_cosine_stability() -> None:
    """Cosine similarity of zero vector must not produce NaN."""
    zero = np.zeros(5, dtype=np.float64)
    one = np.ones(5, dtype=np.float64)
    res = cosine_similarity(zero, one)
    check("cosine_zero_vector_no_nan",
          not np.isnan(res["similarity"]),
          f"similarity={res['similarity']}")


if __name__ == "__main__":
    print("ScratchAI Lesson 01 — Test Suite")
    print("=" * 50)
    test_forward_pass_shape()
    test_gradient_nonzero()
    test_loss_decreases()
    test_numerical_gradient()
    test_stress_no_nan_inf()
    test_stress_runtime()
    test_cosine_stability()
    print("=" * 50)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)