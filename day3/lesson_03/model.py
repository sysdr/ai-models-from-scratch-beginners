"""
model.py — ScratchAI Lesson 01: Matrix Transformer
====================================================
Pure NumPy implementation of 2D linear transformations.
No PyTorch. No sklearn. No magic.

Every function here is PURE: same inputs always produce same outputs.
No hidden global state. All shapes annotated.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ── Point Cloud Generators ────────────────────────────────────────

def get_grid_points(extent: float = 2.5, n: int = 11) -> NDArray[np.float64]:
    """
    Generate a uniform 2D grid of points.

    We create two 1D arrays with np.linspace, then use np.meshgrid
    to produce all (x, y) combinations — this is the vectorized
    equivalent of a nested Python for-loop, but runs in C.

    Parameters
    ----------
    extent : half-width of the grid (grid spans [-extent, +extent])
    n      : number of lines per axis (n=11 → 11×11 = 121 points)

    Returns
    -------
    NDArray shape (n*n, 2): each row is one [x, y] grid point
    """
    axis = np.linspace(-extent, extent, n)
    # meshgrid returns two (n, n) arrays; we interleave them into (n², 2)
    xx, yy = np.meshgrid(axis, axis)
    return np.column_stack([xx.ravel(), yy.ravel()])  # shape: (n*n, 2)


def get_unit_circle(n_points: int = 200) -> NDArray[np.float64]:
    """
    Sample n_points evenly around the unit circle.

    Returns NDArray shape (n_points, 2).
    Used to visualize how M distorts a perfect circle into an ellipse —
    which directly shows the singular values of M.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    # np.cos/sin operate element-wise on the full array — no Python loop
    return np.column_stack([np.cos(theta), np.sin(theta)])


def get_dense_grid_lines(
    extent: float = 2.5, n_lines: int = 21
) -> list[NDArray[np.float64]]:
    """
    Generate a list of line segments for a background grid.

    Each element is shape (2, 2): [[x_start, y_start], [x_end, y_end]].
    We return horizontal and vertical lines separately so the caller
    can transform and plot each segment independently.

    Returns
    -------
    list of NDArray, each shape (2, 2)
    """
    ticks = np.linspace(-extent, extent, n_lines)
    lines: list[NDArray[np.float64]] = []

    for t in ticks:
        # horizontal line: y = t, x ∈ [-extent, extent]
        lines.append(np.array([[-extent, t], [extent, t]], dtype=np.float64))
        # vertical line: x = t, y ∈ [-extent, extent]
        lines.append(np.array([[t, -extent], [t, extent]], dtype=np.float64))

    return lines


# ── Core Transform ────────────────────────────────────────────────

def apply_transform(
    M: NDArray[np.float64],
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply 2×2 linear transformation M to every point in the cloud.

    Shape contract:
      M      : (2, 2)
      points : (N, 2)   ← N points, each with x and y coordinate
      output : (N, 2)

    The math:
      For a single point x (shape (2,)):   y = M @ x
      For N points as row vectors in X:    Y = X @ M.T

    Why X @ M.T and not M @ X?
      Matrix multiply rule: (m,k)@(k,n)→(m,n).
      M is (2,2). X (with points as rows) is (N,2).
      M @ X → (2,2)@(N,2) — inner dims 2≠N → CRASH.
      X @ M.T → (N,2)@(2,2) — inner dims 2==2 → (N,2) ✓
      Algebraically: (X M^T)[i,:] = M x[i], so each row
      of the result is M applied to the corresponding input row.

    Parameters
    ----------
    M      : 2×2 transformation matrix
    points : (N, 2) array of 2D coordinates

    Returns
    -------
    (N, 2) transformed coordinates
    """
    if M.shape != (2, 2):
        raise ValueError(f"M must be shape (2,2), got {M.shape}")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be shape (N,2), got {points.shape}")

    return points @ M.T  # single BLAS dgemm call — the whole lesson


# ── Matrix Properties ─────────────────────────────────────────────

def compute_determinant(M: NDArray[np.float64]) -> float:
    """
    Compute the determinant of a 2×2 matrix using the direct formula.

    For M = [[a, b], [c, d]]:  det(M) = a*d - b*c

    Geometric interpretation:
      |det(M)| = area of the parallelogram spanned by M's columns.
      If det(M) > 0, orientation is preserved (no flip).
      If det(M) < 0, orientation is reversed (mirror).
      If det(M) = 0, M is singular — it collapses 2D space to a line.

    We avoid np.linalg.det here deliberately — the 2×2 formula is
    one multiply-subtract, and making it explicit builds intuition.
    """
    a, b = float(M[0, 0]), float(M[0, 1])
    c, d = float(M[1, 0]), float(M[1, 1])
    return a * d - b * c


def compute_inverse(
    M: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, str]:
    """
    Compute the inverse of a 2×2 matrix using the analytic formula.

    For M = [[a, b], [c, d]]:
      M⁻¹ = (1 / det(M)) * [[d, -b], [-c, a]]

    This formula is derived from the adjugate (classical adjoint):
    swap the diagonal, negate the off-diagonal, divide by det.

    Returns
    -------
    (inv_matrix, status_message)
      inv_matrix is None if M is singular (|det| < epsilon).

    Numerical stability:
      We guard against floating-point near-zero determinants with
      an epsilon threshold of 1e-10 rather than exact zero comparison.
      A matrix whose true det is 0 can compute to 1e-16 due to
      floating-point rounding — exact zero comparison would miss it.
    """
    det = compute_determinant(M)
    EPSILON = 1e-10

    if abs(det) < EPSILON:
        return None, f"Singular matrix — det ≈ {det:.2e}. No inverse exists."

    a, b = float(M[0, 0]), float(M[0, 1])
    c, d = float(M[1, 0]), float(M[1, 1])
    inv = (1.0 / det) * np.array([[d, -b], [-c, a]], dtype=np.float64)
    return inv, "OK"


def compute_transpose(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return the transpose of M: M.T swaps rows and columns.

    M[i, j] → M.T[j, i]

    In NumPy, .T is a zero-copy view — no new memory is allocated.
    We return np.array(M.T) here to produce an explicit (2, 2) copy
    for clean downstream display.

    Geometric meaning: if M is a pure rotation by angle θ,
    then M.T = M⁻¹ (transpose = inverse for orthogonal matrices).
    """
    return np.array(M.T, dtype=np.float64)


def decompose_svd(
    M: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Decompose M via Singular Value Decomposition: M = U @ diag(s) @ Vt

    U  : (2, 2) orthogonal matrix — second rotation
    s  : (2,)   singular values — axis-aligned scaling factors
    Vt : (2, 2) orthogonal matrix — first rotation (already transposed)

    Geometric reading:
      Every 2×2 linear map = rotate (Vt) → scale axes (s) → rotate (U)
      The singular values s[0], s[1] are the semi-axes of the ellipse
      that M maps the unit circle to.
      If either singular value is 0 → matrix is rank-deficient.

    Uses np.linalg.svd from NumPy's LAPACK bindings.
    This is the one linalg call we allow — it is the reference
    decomposition, not a shortcut around the core lesson.
    """
    U, s, Vt = np.linalg.svd(M)
    return U, s, Vt


def condition_number(M: NDArray[np.float64]) -> float:
    """
    Compute the condition number of M: σ_max / σ_min.

    A high condition number (> 1000) means M is nearly singular:
    small perturbations in input produce large perturbations in output.
    This is the numerical stability warning signal for any linear system.
    """
    _, s, _ = decompose_svd(M)
    if s[1] < 1e-12:
        return float("inf")
    return float(s[0] / s[1])


# ── Preset Matrices ───────────────────────────────────────────────

def rotation_matrix(angle_deg: float) -> NDArray[np.float64]:
    """
    Pure rotation by angle_deg degrees (counterclockwise).

    R = [[cos θ, -sin θ],
         [sin θ,  cos θ]]

    det(R) = cos²θ + sin²θ = 1 — rotations never scale area.
    R.T = R⁻¹ — transpose undoes the rotation.
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def shear_matrix(kx: float = 0.5, ky: float = 0.0) -> NDArray[np.float64]:
    """
    Shear matrix: tilts space along x by kx and along y by ky.

    [[1, kx],   ← adds kx * y to every x coordinate
     [ky, 1]]   ← adds ky * x to every y coordinate

    det = 1 - kx*ky. For pure x-shear (ky=0): det = 1 (area preserved).
    """
    return np.array([[1.0, kx], [ky, 1.0]], dtype=np.float64)


def scale_matrix(sx: float = 2.0, sy: float = 0.5) -> NDArray[np.float64]:
    """
    Axis-aligned scaling: stretch x by sx, y by sy.

    [[sx, 0],
     [0, sy]]

    det = sx * sy — product of scale factors.
    """
    return np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)


IDENTITY: NDArray[np.float64] = np.eye(2, dtype=np.float64)

SINGULAR: NDArray[np.float64] = np.array(
    [[1.0, 2.0], [2.0, 4.0]], dtype=np.float64
)  # det = 1*4 - 2*2 = 0 → collapses space to the line y = 2x


PRESETS: dict[str, NDArray[np.float64]] = {
    "Identity":         IDENTITY.copy(),
    "Rotation 45°":     rotation_matrix(45),
    "Rotation 90°":     rotation_matrix(90),
    "Shear X":          shear_matrix(kx=0.8, ky=0.0),
    "Scale (2x, 0.5y)": scale_matrix(2.0, 0.5),
    "Reflection (x-axis)": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "Singular (det=0)": SINGULAR.copy(),
}
    