"""
model.py — Tensor Shape Analyzer
ScratchAI Beginner · Lesson 01: Data as Tensors

Pure NumPy implementation. No PyTorch, no sklearn, no TensorFlow.
Every function is pure (no hidden global state). All types are explicit.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShapeProfile:
    """Complete structural description of a NumPy array."""
    rank: int
    shape: tuple[int, ...]
    strides_bytes: tuple[int, ...]
    dtype: str
    nbytes: int
    n_elements: int
    is_contiguous: bool
    axis_labels: list[str]


@dataclass(frozen=True)
class AxisStats:
    """Per-axis descriptive statistics computed by vectorized reduction."""
    axis: int
    label: str
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    minimum: NDArray[np.float64]
    maximum: NDArray[np.float64]
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class ReshapeCandidate:
    """A valid (rows, cols) pair for 2-D reshape of an array."""
    rows: int
    cols: int
    is_square: bool


@dataclass(frozen=True)
class SliceResult:
    """A 2-D cross-section extracted from a higher-rank tensor."""
    data: NDArray[np.float64]
    origin_shape: tuple[int, ...]
    slice_axes: tuple[int, int]
    fixed_indices: dict[int, int]


# ─── Rank Inference ───────────────────────────────────────────────────────────

_AXIS_LABELS: dict[int, list[str]] = {
    0: ["scalar"],
    1: ["elements"],
    2: ["rows (samples)", "cols (features)"],
    3: ["batch", "rows (height)", "cols (width)"],
    4: ["batch", "height", "width", "channels"],
}


def infer_rank(arr: NDArray) -> tuple[int, list[str]]:
    """
    Return (rank, axis_label_list) for any ndarray up to rank 4.

    Labels follow the deep-learning convention:
      axis 0 → batch / sample dimension
      axis -1 → feature / channel dimension

    Parameters
    ----------
    arr : NDArray
        Any NumPy array of rank 0–4.

    Returns
    -------
    rank : int
        Number of dimensions (arr.ndim).
    labels : list[str]
        Human-readable name for each axis.
    """
    rank = int(arr.ndim)
    labels = _AXIS_LABELS.get(rank, [f"axis_{i}" for i in range(rank)])
    return rank, labels


# ─── Shape Profile ────────────────────────────────────────────────────────────

def compute_shape_stats(arr: NDArray) -> ShapeProfile:
    """
    Build a complete ShapeProfile for *arr* without modifying it.

    Memory footprint = n_elements × itemsize.
    Contiguity check uses arr.flags['C_CONTIGUOUS']: non-contiguous arrays
    (e.g., after transpose) produce strides that skip bytes — reshaping
    them forces a silent data copy inside NumPy.

    Parameters
    ----------
    arr : NDArray
        Input array of any rank, dtype, or memory layout.

    Returns
    -------
    ShapeProfile
        Frozen dataclass with all structural metadata.
    """
    rank, labels = infer_rank(arr)
    return ShapeProfile(
        rank=rank,
        shape=tuple(arr.shape),
        strides_bytes=tuple(arr.strides),
        dtype=str(arr.dtype),
        nbytes=int(arr.nbytes),
        n_elements=int(arr.size),
        is_contiguous=bool(arr.flags["C_CONTIGUOUS"]),
        axis_labels=labels,
    )


# ─── Axis Statistics ──────────────────────────────────────────────────────────

def compute_axis_stats(arr: NDArray, axis: int) -> AxisStats:
    """
    Compute mean, std, min, max by reducing along *axis* in one vectorized pass.

    Why not a Python loop?
    ----------------------
    For a (1000, 512) array, computing column means with a loop:
      [arr[:, j].mean() for j in range(512)]
    allocates 512 temporary Python float objects and calls the Python
    interpreter 512 times. np.mean(arr, axis=0) allocates one (512,) array
    and runs a single C-level loop. Same arithmetic; one Python call total.

    Parameters
    ----------
    arr : NDArray
        Array of rank ≥ 1.
    axis : int
        Axis along which to reduce. 0 → reduce over samples (column stats).
                                   -1 → reduce over features (row stats).

    Returns
    -------
    AxisStats
        Reduction output shape is arr.shape with *axis* removed.
    """
    _, labels = infer_rank(arr)
    ax_label = labels[axis] if abs(axis) < len(labels) else f"axis_{axis}"

    # All four reductions in four vectorized calls — no Python loops.
    mean_val: NDArray[np.float64] = np.mean(arr, axis=axis).astype(np.float64)
    std_val: NDArray[np.float64]  = np.std(arr,  axis=axis).astype(np.float64)
    min_val: NDArray[np.float64]  = np.min(arr,  axis=axis).astype(np.float64)
    max_val: NDArray[np.float64]  = np.max(arr,  axis=axis).astype(np.float64)

    return AxisStats(
        axis=axis,
        label=ax_label,
        mean=mean_val,
        std=std_val,
        minimum=min_val,
        maximum=max_val,
        output_shape=tuple(mean_val.shape),
    )


# ─── Reshape Validation & Candidates ─────────────────────────────────────────

def validate_reshape(arr: NDArray, new_shape: tuple[int, ...]) -> tuple[bool, str]:
    """
    Check whether *arr* can be reshaped to *new_shape* without errors.

    NumPy reshape requires: product(new_shape) == arr.size.
    A -1 in new_shape is allowed: NumPy infers that axis.

    This function replicates the check NumPy performs internally so the app
    can surface the error message BEFORE calling reshape — enabling live
    UI feedback without try/except overhead in the render loop.

    Parameters
    ----------
    arr : NDArray
        Source array.
    new_shape : tuple[int, ...]
        Desired output shape. May contain at most one -1.

    Returns
    -------
    (is_valid, message) : tuple[bool, str]
        is_valid=True if reshape will succeed; message explains why if not.
    """
    n = int(arr.size)

    # Count inference axes
    infer_axes = [s for s in new_shape if s == -1]
    if len(infer_axes) > 1:
        return False, "At most one dimension in new_shape can be -1."

    known = [s for s in new_shape if s != -1]
    if any(s <= 0 for s in known):
        return False, f"All explicit dimensions must be > 0. Got: {new_shape}"

    known_product = int(np.prod(known)) if known else 1

    if infer_axes:
        if n % known_product != 0:
            return (
                False,
                f"Cannot infer dimension: {n} elements ÷ {known_product} = "
                f"{n / known_product:.4f} (not an integer).",
            )
        return True, f"Valid. Inferred dimension = {n // known_product}."

    new_product = int(np.prod(new_shape))
    if new_product != n:
        return (
            False,
            f"Shape mismatch: array has {n} elements but "
            f"{new_shape} requires {new_product}. "
            f"Difference: {abs(new_product - n)} element(s).",
        )
    return True, f"Valid. {n} elements → {new_shape}."


def generate_reshape_candidates(arr: NDArray) -> list[ReshapeCandidate]:
    """
    Return all (rows, cols) pairs that exactly tile *arr* into a 2-D grid.

    Algorithm: find every divisor of arr.size up to sqrt(arr.size).
    For each divisor d, (d, size/d) is a valid candidate.
    Complexity: O(√n) — fast even for n = 10^7.

    This is a pure Python loop over a small integer search space (at most
    ~100 iterations for typical tensor sizes), not an array operation —
    correct use of Python vs NumPy: use Python for control flow over
    small integers, NumPy for operations over large arrays.

    Parameters
    ----------
    arr : NDArray
        Source array of any rank.

    Returns
    -------
    list[ReshapeCandidate]
        Sorted by rows ascending. Includes the (1, n) and (n, 1) trivial cases.
    """
    n = int(arr.size)
    candidates: list[ReshapeCandidate] = []
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            candidates.append(ReshapeCandidate(rows=d, cols=n // d, is_square=(d == n // d)))
            if d != n // d:
                candidates.append(ReshapeCandidate(rows=n // d, cols=d, is_square=False))
    return sorted(candidates, key=lambda c: c.rows)


# ─── Slice Extraction ─────────────────────────────────────────────────────────

def extract_slice(
    arr: NDArray,
    slice_axes: tuple[int, int] = (0, 1),
    fixed_indices: dict[int, int] | None = None,
) -> SliceResult:
    """
    Extract a 2-D cross-section from *arr* by fixing all axes except *slice_axes*.

    For a 4-D array of shape (B, H, W, C):
      extract_slice(arr, slice_axes=(1, 2), fixed_indices={0: 0, 3: 0})
    returns arr[0, :, :, 0] — the first channel of the first batch item.

    For a 2-D array, slice_axes=(0,1) returns the full array.

    The indexing is vectorized: np.take-style advanced indexing, no Python
    element loops. Shape of output is always 2-D.

    Parameters
    ----------
    arr : NDArray
        Input array of rank 2–4.
    slice_axes : tuple[int, int]
        Two axes to keep as the 2-D slice dimensions.
    fixed_indices : dict[int, int] | None
        For every axis NOT in slice_axes, the index to fix.
        Defaults to 0 for each missing axis.

    Returns
    -------
    SliceResult
        .data is a 2-D float64 array.
    """
    rank = arr.ndim
    if rank < 2:
        # Pad 1-D arrays to (1, n) for uniform display
        return SliceResult(
            data=arr.reshape(1, -1).astype(np.float64),
            origin_shape=tuple(arr.shape),
            slice_axes=(0, 1),
            fixed_indices={},
        )

    fixed = fixed_indices or {}
    # Build a full index tuple: slice for kept axes, integer for fixed axes
    idx: list[int | slice] = []
    for ax in range(rank):
        if ax in slice_axes:
            idx.append(slice(None))       # keep this axis
        else:
            idx.append(fixed.get(ax, 0))  # fix this axis at given index

    sliced = arr[tuple(idx)]

    # After advanced indexing, ensure 2-D output
    match sliced.ndim:
        case 2:
            data_2d = sliced.astype(np.float64)
        case 1:
            data_2d = sliced.reshape(1, -1).astype(np.float64)
        case _:
            # Flatten all but last two dimensions if somehow >2D remain
            data_2d = sliced.reshape(-1, sliced.shape[-1]).astype(np.float64)

    return SliceResult(
        data=data_2d,
        origin_shape=tuple(arr.shape),
        slice_axes=slice_axes,
        fixed_indices={k: v for k, v in fixed.items() if k not in slice_axes},
    )


# ─── NaN / Inf Audit ─────────────────────────────────────────────────────────

def audit_array(arr: NDArray) -> dict[str, int | float | bool]:
    """
    Run a vectorized health-check on *arr*.

    All operations are single-pass reductions — no Python element loops.
    np.isnan and np.isinf produce boolean arrays in C; .sum() reduces them
    to scalars in one vectorized call.

    Returns
    -------
    dict with keys: nan_count, inf_count, zero_count, value_min,
                    value_max, value_range, has_issues.
    """
    flat = arr.ravel().astype(np.float64)
    nan_count  = int(np.isnan(flat).sum())
    inf_count  = int(np.isinf(flat).sum())
    zero_count = int((flat == 0.0).sum())

    finite = flat[np.isfinite(flat)]
    if finite.size > 0:
        v_min   = float(finite.min())
        v_max   = float(finite.max())
        v_range = v_max - v_min
    else:
        v_min = v_max = v_range = float("nan")

    return {
        "nan_count":  nan_count,
        "inf_count":  inf_count,
        "zero_count": zero_count,
        "value_min":  v_min,
        "value_max":  v_max,
        "value_range": v_range,
        "has_issues": bool(nan_count > 0 or inf_count > 0),
    }


# ─── Synthetic Tensor Generator ───────────────────────────────────────────────

def make_demo_tensor(rank: int, seed: int = 42) -> NDArray[np.float64]:
    """
    Generate a demo tensor of the given rank for the Streamlit app.

    Shapes follow real-world conventions:
      0D → scalar
      1D → (128,)         — feature vector
      2D → (32, 16)       — small batch of feature vectors
      3D → (8, 28, 28)    — batch of grayscale images (MNIST-like)
      4D → (4, 16, 16, 3) — batch of tiny RGB images

    Uses a fixed seed for reproducibility; the app passes seed=slider_value
    to make the demo interactive.
    """
    rng = np.random.default_rng(seed)
    match rank:
        case 0: return np.float64(rng.standard_normal())
        case 1: return rng.standard_normal(128).astype(np.float64)
        case 2: return rng.standard_normal((32, 16)).astype(np.float64)
        case 3: return rng.standard_normal((8, 28, 28)).astype(np.float64)
        case 4: return rng.standard_normal((4, 16, 16, 3)).astype(np.float64)
        case _: raise ValueError(f"rank must be 0–4, got {rank}")
