"""
ScratchAI-Beginner | Lesson 01: NumPy Playground
Core computation module — pure NumPy, no ML frameworks.
"""

from __future__ import annotations
import time
import numpy as np
from numpy.typing import NDArray

# ── Vector Operations ────────────────────────────────────────────────────

def dot_product(a: NDArray[np.float64], b: NDArray[np.float64]) -> dict:
    """
    Compute the dot product of two 1-D vectors and return a full
    step-by-step breakdown.

    The dot product is the sum of element-wise products:
        a · b = Σ a_i * b_i

    This is the atomic operation behind every neuron activation:
        output = weights · inputs + bias
    """
    assert a.ndim == 1 and b.ndim == 1, "Inputs must be 1-D vectors."
    assert a.shape == b.shape, (
        f"Shape mismatch: {a.shape} vs {b.shape}. "
        "Dot product requires equal-length vectors."
    )
    elementwise = a * b          # shape: (n,)  ← broadcast-free multiply
    result = np.sum(elementwise) # shape: ()    ← scalar reduction
    return {
        "a": a,
        "b": b,
        "elementwise_products": elementwise,
        "result": float(result),
        "operation": "dot",
    }


def outer_product(a: NDArray[np.float64], b: NDArray[np.float64]) -> dict:
    """
    Compute the outer product of two vectors.

    a ⊗ b produces a matrix M where M[i,j] = a[i] * b[j].
    Shape: (len(a),) ⊗ (len(b),) → (len(a), len(b))

    NumPy idiom: np.outer(a, b) == a[:, np.newaxis] * b[np.newaxis, :]
    The newaxis insertion reshapes a to (n,1) and b to (1,m), then
    broadcasting produces the full (n,m) matrix without any Python loop.
    """
    assert a.ndim == 1 and b.ndim == 1
    # Explicit broadcast form — pedagogically clearer than np.outer
    M = a[:, np.newaxis] * b[np.newaxis, :]
    return {
        "a": a,
        "b": b,
        "result": M,
        "shape": M.shape,
        "operation": "outer",
    }


def cosine_similarity(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> dict:
    """
    Compute cosine similarity between two vectors, numerically stabilized.

    cos(θ) = (a · b) / (‖a‖ · ‖b‖)

    Stability: if either vector has near-zero norm, we clip the denominator
    to eps=1e-8 to avoid NaN from division-by-zero.
    """
    eps = 1e-8
    dot = np.dot(a.astype(np.float64), b.astype(np.float64))
    norm_a = np.maximum(np.linalg.norm(a), eps)
    norm_b = np.maximum(np.linalg.norm(b), eps)
    similarity = dot / (norm_a * norm_b)
    return {
        "a": a,
        "b": b,
        "dot": float(dot),
        "norm_a": float(norm_a),
        "norm_b": float(norm_b),
        "similarity": float(np.clip(similarity, -1.0, 1.0)),
        "operation": "cosine",
    }


# ── Matrix Operations ────────────────────────────────────────────────────

def matrix_multiply(
    A: NDArray[np.float64], B: NDArray[np.float64]
) -> dict:
    """
    Matrix multiplication: C = A @ B

    Shape contract:  (m, k) @ (k, n) → (m, n)
    The inner dimensions MUST match. NumPy raises ValueError otherwise.

    This is the forward pass of a fully-connected layer:
        output = input @ weights.T + bias
    """
    assert A.ndim == 2 and B.ndim == 2, "Both inputs must be 2-D matrices."
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Shape mismatch for matmul: {A.shape} @ {B.shape}. "
            f"A.shape[1]={A.shape[1]} must equal B.shape[0]={B.shape[0]}."
        )
    C = A @ B
    return {
        "A": A,
        "B": B,
        "result": C,
        "shape_A": A.shape,
        "shape_B": B.shape,
        "shape_C": C.shape,
        "operation": "matmul",
    }


def broadcasting_demo(
    A: NDArray[np.float64], b: NDArray[np.float64]
) -> dict:
    """
    Demonstrate NumPy broadcasting: add a bias vector b to every row of A.

    A has shape (m, n). b has shape (n,).
    NumPy right-aligns shapes: (m, n) vs (1, n) [after prepending 1].
    Dimension 0: m vs 1 → stretch b across all m rows (zero-stride trick).
    Dimension 1: n vs n → match.
    Result shape: (m, n). No data copied — just stride manipulation.

    Failure mode: if b has shape (m,) instead of (n,), NumPy sees
    (m, n) vs (1, m) which fails when n ≠ m.
    """
    assert A.ndim == 2, "A must be a 2-D matrix."
    assert b.ndim == 1, "b must be a 1-D vector."
    steps: list[str] = [
        f"A.shape = {A.shape}",
        f"b.shape = {b.shape}  →  right-aligned: (1, {b.shape[0]})",
        f"Broadcasting rule: ({A.shape[0]}, {A.shape[1]}) + (1, {b.shape[0]})",
    ]
    if b.shape[0] != A.shape[1]:
        raise ValueError(
            f"Cannot broadcast: A has {A.shape[1]} columns but "
            f"b has {b.shape[0]} elements. "
            f"Shapes (m={A.shape[0]}, n={A.shape[1]}) + (n={b.shape[0]},) require n to match."
        )
    result = A + b   # broadcasting: (m,n) + (n,) → (m,n)
    steps.append(f"Result.shape = {result.shape}  ✓")
    return {
        "A": A,
        "b": b,
        "result": result,
        "steps": steps,
        "operation": "broadcast_add",
    }


def dtype_cast_demo(values: list[int | float], target_dtype: str) -> dict:
    """
    Show how dtype affects the values stored in a NumPy array.

    Integer dtypes truncate fractional parts.
    int8/int16/int32 overflow silently (wraps around modulo 2^N).
    float32 loses precision for large integers (> 2^24).
    Always cast to float64 before operations requiring precision.
    """
    supported = {
        "int8": np.int8, "int16": np.int16,
        "int32": np.int32, "int64": np.int64,
        "float32": np.float32, "float64": np.float64,
    }
    assert target_dtype in supported, f"Unknown dtype: {target_dtype}"
    original = np.array(values, dtype=np.float64)
    casted = original.astype(supported[target_dtype])
    precision_lost = not np.allclose(
        original, casted.astype(np.float64), rtol=1e-5
    )
    overflow_detected = bool(
        np.any(casted.astype(np.float64) != np.round(original))
        and "int" in target_dtype
    )
    return {
        "original": original,
        "casted": casted,
        "original_dtype": str(original.dtype),
        "target_dtype": target_dtype,
        "precision_lost": precision_lost,
        "overflow_detected": overflow_detected,
        "operation": "dtype_cast",
    }


def vectorization_benchmark(size: int = 500_000) -> dict:
    """
    Compare loop vs vectorized element-wise multiply on size elements.

    Loop: Python iterates size times — each iteration allocates a Python
    float object, checks types, and dispatches through the interpreter.
    Vectorized: np.multiply calls a single C function that runs a tight
    SIMD loop over contiguous float64 memory. Same math, different execution.
    """
    rng = np.random.default_rng(42)
    a = rng.random(size, dtype=np.float64)
    b = rng.random(size, dtype=np.float64)

    # Loop version
    t0 = time.perf_counter()
    loop_result = np.empty(size, dtype=np.float64)
    for i in range(size):
        loop_result[i] = a[i] * b[i]
    loop_time = time.perf_counter() - t0

    # Vectorized version
    t0 = time.perf_counter()
    vec_result = a * b
    vec_time = time.perf_counter() - t0

    max_diff = float(np.max(np.abs(loop_result - vec_result)))

    return {
        "size": size,
        "loop_time_ms": loop_time * 1000,
        "vec_time_ms": vec_time * 1000,
        "speedup": loop_time / max(vec_time, 1e-9),
        "max_diff": max_diff,
        "identical": max_diff < 1e-14,
        "operation": "benchmark",
    }


def simulate_shape_error() -> dict:
    """
    Intentionally trigger a shape mismatch in matrix multiply.
    Returns the error message rather than raising, so the UI can display it.
    """
    A = np.ones((3, 4))
    B = np.ones((3, 4))   # wrong: should be (4, N)
    try:
        _ = A @ B
        return {"error": None, "message": "No error (unexpected)"}
    except ValueError as e:
        return {
            "error": "ValueError",
            "message": str(e),
            "A_shape": A.shape,
            "B_shape": B.shape,
            "hint": (
                "matmul requires A.shape[1] == B.shape[0]. "
                f"Here A.shape[1]={A.shape[1]} ≠ B.shape[0]={B.shape[0]}."
            ),
        }


def simulate_dtype_overflow() -> dict:
    """
    Intentionally trigger int8 overflow — the silent bug.
    int8 holds values in [-128, 127]. Adding 100 + 100 = 200 > 127.
    NumPy wraps around: 200 mod 256 - 128 = -56. No warning.
    """
    a = np.array([100, 127, 50], dtype=np.int8)
    b = np.array([100, 1, 200], dtype=np.int8)
    result = a + b
    expected = a.astype(np.int64) + b.astype(np.int64)
    return {
        "a": a,
        "b": b,
        "result_int8": result,
        "expected_int64": expected,
        "overflow_mask": result.astype(np.int64) != expected,
        "hint": (
            "int8 wraps silently. "
            "127 + 1 = -128 in int8. "
            "Cast to float64 before arithmetic."
        ),
    }