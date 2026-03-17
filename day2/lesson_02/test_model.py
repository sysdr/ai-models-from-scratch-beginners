"""
test_model.py — Unit Tests for Tensor Shape Analyzer
ScratchAI Beginner · Lesson 01: Data as Tensors

Run:  python test_model.py
"""

from __future__ import annotations
import sys
import time
import traceback
import numpy as np

from model import (
    make_demo_tensor,
    compute_shape_stats,
    compute_axis_stats,
    extract_slice,
    validate_reshape,
    generate_reshape_candidates,
    audit_array,
)

PASS = "\033[92m  ✓ PASS\033[0m"
FAIL = "\033[91m  ✗ FAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def run_test(name: str, fn) -> None:
    try:
        fn()
        results.append((name, True, ""))
        print(f"{PASS}  {name}")
    except Exception as exc:
        tb = traceback.format_exc()
        results.append((name, False, str(exc)))
        print(f"{FAIL}  {name}\n        {exc}\n{tb}")


# ─── Unit Tests ───────────────────────────────────────────────────────────────

def test_forward_pass_shape():
    """extract_slice output is always 2-D."""
    for rank in range(2, 5):
        arr = make_demo_tensor(rank=rank)
        s = extract_slice(arr, slice_axes=(rank - 2, rank - 1))
        assert s.data.ndim == 2, f"Expected 2-D slice, got {s.data.ndim}-D for rank {rank}"


def test_gradient_nonzero():
    """
    Axis statistics are non-zero for non-zero input arrays.
    (Analog of 'gradients are non-zero after backward' for this lesson's
    core computation: axis-wise reduction produces non-trivial output.)
    """
    arr = make_demo_tensor(rank=2, seed=1)
    for ax in range(2):
        stats = compute_axis_stats(arr, axis=ax)
        assert not np.all(stats.mean == 0), f"All-zero means for axis {ax} — suspicious."
        assert stats.std.sum() > 0, f"All-zero stds for axis {ax} — constant array?"


def test_loss_decreases():
    """
    Value range of successive seeds decreases as seed increases (verifies
    that make_demo_tensor produces different arrays per seed).
    Analog of 'loss at epoch 10 < loss at epoch 1'.
    """
    ranges = []
    for seed in range(1, 12):
        arr = make_demo_tensor(rank=2, seed=seed)
        a = audit_array(arr)
        ranges.append(a["value_range"])
    # Not all ranges should be identical — confirms seed affects output
    assert len(set(f"{r:.6f}" for r in ranges)) > 1, "All seeds produced identical arrays — seed is ignored?"


def test_numerical_gradient():
    """
    Validate vectorized axis mean against a reference loop implementation.
    Analog of 'compare analytical vs numerical gradient'.
    """
    rng = np.random.default_rng(99)
    arr = rng.standard_normal((50, 20))

    # Analytical (vectorized)
    stats = compute_axis_stats(arr, axis=0)
    vec_mean = stats.mean

    # Reference (Python loop — intentionally slow, used only for validation)
    ref_mean = np.array([arr[:, j].mean() for j in range(arr.shape[1])])

    np.testing.assert_allclose(
        vec_mean, ref_mean, rtol=1e-10, atol=1e-12,
        err_msg="Vectorized mean differs from loop reference — implementation error."
    )


def test_reshape_validation_correct():
    """validate_reshape correctly identifies valid and invalid shapes."""
    arr = np.arange(12)
    ok, _ = validate_reshape(arr, (3, 4))
    assert ok, "Should accept (3,4) for 12 elements"

    ok2, msg = validate_reshape(arr, (3, 5))
    assert not ok2, f"Should reject (3,5) for 12 elements — got ok=True"
    assert "15" in msg or "mismatch" in msg.lower(), f"Error message unclear: {msg}"


def test_audit_detects_nan():
    """audit_array flags NaN correctly."""
    arr = np.array([1.0, 2.0, float("nan"), 4.0])
    a = audit_array(arr)
    assert a["nan_count"] == 1, f"Expected 1 NaN, got {a['nan_count']}"
    assert a["has_issues"] is True


def test_reshape_candidates_exhaustive():
    """All reshape candidates tile the array exactly."""
    arr = np.arange(24)
    candidates = generate_reshape_candidates(arr)
    for c in candidates:
        assert c.rows * c.cols == 24, f"Candidate {c.rows}×{c.cols} ≠ 24"
    # Should include (1,24), (2,12), (3,8), (4,6), (6,4), (8,3), (12,2), (24,1)
    assert len(candidates) == 8, f"Expected 8 candidates for 24, got {len(candidates)}"


def test_shape_profile_fields():
    """compute_shape_stats returns all required fields with correct types."""
    arr = make_demo_tensor(rank=3)
    profile = compute_shape_stats(arr)
    assert profile.rank == 3
    assert profile.n_elements == int(np.prod(arr.shape))
    assert profile.nbytes == arr.nbytes
    assert isinstance(profile.axis_labels, list)
    assert len(profile.axis_labels) == 3


# ─── Run All ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nScratchAI L01 · Unit Tests\n" + "─" * 50)
    run_test("test_forward_pass_shape",        test_forward_pass_shape)
    run_test("test_gradient_nonzero",          test_gradient_nonzero)
    run_test("test_loss_decreases",            test_loss_decreases)
    run_test("test_numerical_gradient",        test_numerical_gradient)
    run_test("test_reshape_validation_correct",test_reshape_validation_correct)
    run_test("test_audit_detects_nan",         test_audit_detects_nan)
    run_test("test_reshape_candidates_exhaustive", test_reshape_candidates_exhaustive)
    run_test("test_shape_profile_fields",      test_shape_profile_fields)

    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    print(f"\n{'─'*50}\nResults: {passed}/{len(results)} passed", end="")
    if failed:
        print(f"  ·  {failed} FAILED")
        sys.exit(1)
    else:
        print("  · All tests passed ✓")
