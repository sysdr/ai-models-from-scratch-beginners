"""
test_model.py — Unit Tests for ScratchAI Lesson 01
Run: python test_model.py
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

from model import (
    IDENTITY,
    PRESETS,
    apply_transform,
    compute_determinant,
    compute_inverse,
    compute_transpose,
    condition_number,
    decompose_svd,
    get_grid_points,
    get_unit_circle,
    rotation_matrix,
)

PASS = "✓ PASS"
FAIL = "✗ FAIL"
results: list[tuple[str, str, str]] = []


def test(name: str, fn):
    try:
        fn()
        results.append((name, PASS, ""))
    except Exception as e:
        results.append((name, FAIL, str(e)))


# ── Unit Tests ────────────────────────────────────────────────────

def test_forward_pass_shape():
    """apply_transform output shape must match input shape."""
    M = rotation_matrix(45)
    pts = get_grid_points(extent=2.0, n=10)  # (100, 2)
    out = apply_transform(M, pts)
    assert out.shape == pts.shape, f"Expected {pts.shape}, got {out.shape}"


def test_identity_is_noop():
    """Applying the identity matrix must return points unchanged."""
    pts = get_unit_circle(50)
    out = apply_transform(IDENTITY, pts)
    diff = np.max(np.abs(out - pts))
    assert diff < 1e-12, f"Max deviation from identity: {diff:.2e}"


def test_determinant_rotation():
    """Rotation matrices have determinant exactly 1."""
    for angle in [0, 30, 45, 90, 135, 180, 270]:
        R = rotation_matrix(angle)
        det = compute_determinant(R)
        assert abs(det - 1.0) < 1e-10, f"R({angle}°) det={det:.6f} ≠ 1"


def test_determinant_singular():
    """The singular preset matrix must have determinant ≈ 0."""
    det = compute_determinant(PRESETS["Singular (det=0)"])
    assert abs(det) < 1e-10, f"Expected det≈0, got {det}"


def test_inverse_round_trip():
    """M @ M⁻¹ must equal the identity matrix (residual < 1e-10)."""
    for name, M in PRESETS.items():
        if name == "Singular (det=0)":
            continue
        inv_M, status = compute_inverse(M)
        assert status == "OK", f"{name}: inverse failed: {status}"
        residual = np.max(np.abs(M @ inv_M - np.eye(2)))
        assert residual < 1e-10, (
            f"{name}: M @ M⁻¹ residual={residual:.2e}"
        )


def test_singular_inverse_returns_none():
    """compute_inverse on a singular matrix must return (None, message)."""
    inv_M, status = compute_inverse(PRESETS["Singular (det=0)"])
    assert inv_M is None, "Expected None for singular inverse"
    assert "Singular" in status or "det" in status


def test_transpose_shape():
    """Transpose of a (2,2) matrix must be (2,2)."""
    M = np.array([[1, 2], [3, 4]], dtype=np.float64)
    assert compute_transpose(M).shape == (2, 2)


def test_transpose_orthogonal():
    """For rotation matrices: M.T @ M == Identity."""
    R = rotation_matrix(37)
    RT_R = compute_transpose(R) @ R
    residual = np.max(np.abs(RT_R - np.eye(2)))
    assert residual < 1e-10, f"R.T @ R residual: {residual:.2e}"


def test_svd_reconstruction():
    """SVD decomposition must reconstruct M exactly: U @ diag(s) @ Vt == M."""
    M = PRESETS["Shear X"]
    U, s, Vt = decompose_svd(M)
    M_reconstructed = U @ np.diag(s) @ Vt
    residual = np.max(np.abs(M_reconstructed - M))
    assert residual < 1e-12, f"SVD reconstruction error: {residual:.2e}"


def test_condition_number_identity():
    """Condition number of identity must be 1.0."""
    cond = condition_number(IDENTITY)
    assert abs(cond - 1.0) < 1e-10, f"cond(I) = {cond}"


def test_no_nan_inf_in_transforms():
    """All preset transforms applied to random inputs must produce finite output."""
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((500, 2))
    for name, M in PRESETS.items():
        if name == "Singular (det=0)":
            continue
        out = apply_transform(M, pts)
        assert np.all(np.isfinite(out)), f"{name}: non-finite output"


def test_transform_linearity():
    """
    Linear map property: M(a + b) == M(a) + M(b).
    Tests that apply_transform is genuinely linear.
    """
    M   = rotation_matrix(60)
    rng = np.random.default_rng(7)
    a   = rng.standard_normal((100, 2))
    b   = rng.standard_normal((100, 2))

    lhs = apply_transform(M, a + b)
    rhs = apply_transform(M, a) + apply_transform(M, b)
    assert np.allclose(lhs, rhs, atol=1e-12), "Linearity violated"


# ── Stress Test ───────────────────────────────────────────────────

def test_stress_1000_batches():
    """
    1000 forward passes with random batch sizes and random matrices.
    All outputs must be finite. Total runtime must be < 5 seconds.
    """
    rng   = np.random.default_rng(0)
    t0    = time.perf_counter()
    M_ref = rotation_matrix(30)

    for i in range(1000):
        n_pts = rng.integers(1, 500)
        pts   = rng.standard_normal((n_pts, 2))
        # Random well-conditioned matrix (not singular by construction)
        angle = rng.uniform(0, 360)
        M     = rotation_matrix(angle)
        out   = apply_transform(M, pts)
        assert np.all(np.isfinite(out)), f"NaN/Inf at iteration {i}"

    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Stress test too slow: {elapsed:.2f}s"
    return elapsed


# ── Runner ────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'━'*60}")
    print("  ScratchAI Lesson 01 — Test Suite")
    print(f"{'━'*60}\n")

    test("forward_pass_shape",        test_forward_pass_shape)
    test("identity_is_noop",          test_identity_is_noop)
    test("determinant_rotation",      test_determinant_rotation)
    test("determinant_singular",      test_determinant_singular)
    test("inverse_round_trip",        test_inverse_round_trip)
    test("singular_inverse_none",     test_singular_inverse_returns_none)
    test("transpose_shape",           test_transpose_shape)
    test("transpose_orthogonal",      test_transpose_orthogonal)
    test("svd_reconstruction",        test_svd_reconstruction)
    test("condition_number_identity", test_condition_number_identity)
    test("no_nan_inf",                test_no_nan_inf_in_transforms)
    test("linearity",                 test_transform_linearity)
    test("stress_1000_batches",       test_stress_1000_batches)

    passed = sum(1 for _, r, _ in results if r == PASS)
    total  = len(results)

    for name, result, msg in results:
        status_icon = "✅" if result == PASS else "❌"
        line = f"  {status_icon} {name:<40} {result}"
        if msg:
            line += f"\n      └─ {msg}"
        print(line)

    print(f"\n{'━'*60}")
    print(f"  {passed}/{total} tests passed")
    print(f"{'━'*60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
    