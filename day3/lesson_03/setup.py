#!/usr/bin/env python3.11
"""
ScratchAI-Beginner | Lesson 03: Matrix Transformer
Run: python setup.py
This script regenerates the lesson_03/ workspace (current directory).
"""

import os
import sys
import textwrap
from pathlib import Path


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strip common leading indent (dedent can leave indent if first line is minimal)
    lines = content.split("\n")
    min_indent = None
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = indent if min_indent is None else min(min_indent, indent)
    if min_indent is None:
        min_indent = 0
    stripped = "\n".join(
        line[min_indent:] if len(line) >= min_indent else line for line in lines
    )
    path.write_text(stripped.lstrip(), encoding="utf-8")
    print(f"  ✓ {path}")


def main() -> None:
    base = Path(__file__).resolve().parent
    print(f"\n🔧 Generating ScratchAI Lesson 03 → {base}/\n")

    # ──────────────────────────────────────────────────────────────────────
    # requirements.txt
    # ──────────────────────────────────────────────────────────────────────
    write(base / "requirements.txt", """
        numpy>=1.26
        streamlit>=1.32
        plotly>=5.20
    """)

    # ──────────────────────────────────────────────────────────────────────
    # model.py
    # ──────────────────────────────────────────────────────────────────────
    write(base / "model.py", '''
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
    ''')

    # ──────────────────────────────────────────────────────────────────────
    # app.py
    # ──────────────────────────────────────────────────────────────────────
    write(base / "app.py", '''
        """
        app.py — ScratchAI Lesson 01: Matrix Transformer
        =================================================
        Streamlit web app: type a 2×2 matrix, watch 2D space transform live.
        Launch: streamlit run app.py
        """

        from __future__ import annotations

        import numpy as np
        import plotly.graph_objects as go
        import streamlit as st

        from model import (
            IDENTITY,
            PRESETS,
            apply_transform,
            compute_determinant,
            compute_inverse,
            compute_transpose,
            condition_number,
            decompose_svd,
            get_dense_grid_lines,
            get_unit_circle,
        )

        # ── Page Config ───────────────────────────────────────────────────

        st.set_page_config(
            page_title="Matrix Transformer | ScratchAI L01",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🔷 Matrix Transformer")
        st.caption("ScratchAI — Lesson 01: Linear Algebra in Code")

        # ── Session State Init ────────────────────────────────────────────

        if "matrix" not in st.session_state:
            st.session_state["matrix"] = IDENTITY.copy()

        if "error_mode" not in st.session_state:
            st.session_state["error_mode"] = False

        # ── Sidebar: Controls ─────────────────────────────────────────────

        with st.sidebar:
            st.header("⚙️ Transformation Matrix")
            st.markdown("Edit the 2×2 matrix **M** that transforms 2D space.")

            # Preset selector
            preset_name = st.selectbox(
                "Load Preset",
                options=list(PRESETS.keys()),
                index=0,
            )
            if st.button("Load Preset →"):
                st.session_state["matrix"] = PRESETS[preset_name].copy()
                st.session_state["error_mode"] = False
                st.rerun()

            st.divider()

            # Manual 2×2 entry
            M_current: np.ndarray = st.session_state["matrix"]

            st.markdown("**Row 1**")
            col1, col2 = st.columns(2)
            a = col1.number_input("M[0,0]", value=float(M_current[0, 0]),
                                  step=0.1, format="%.3f", key="m00")
            b = col2.number_input("M[0,1]", value=float(M_current[0, 1]),
                                  step=0.1, format="%.3f", key="m01")

            st.markdown("**Row 2**")
            col3, col4 = st.columns(2)
            c = col3.number_input("M[1,0]", value=float(M_current[1, 0]),
                                  step=0.1, format="%.3f", key="m10")
            d = col4.number_input("M[1,1]", value=float(M_current[1, 1]),
                                  step=0.1, format="%.3f", key="m11")

            M = np.array([[a, b], [c, d]], dtype=np.float64)
            st.session_state["matrix"] = M

            st.divider()

            # Action buttons
            col_err, col_rst = st.columns(2)
            if col_err.button("💥 Simulate Error", use_container_width=True):
                st.session_state["matrix"] = np.array(
                    [[1.0, 2.0], [2.0, 4.0]], dtype=np.float64
                )
                st.session_state["error_mode"] = True
                st.rerun()

            if col_rst.button("↩ Reset", use_container_width=True):
                st.session_state["matrix"] = IDENTITY.copy()
                st.session_state["error_mode"] = False
                st.rerun()

            st.divider()

            # Display options
            st.subheader("👁 Display Options")
            show_original = st.checkbox("Show original grid", value=True)
            show_circle   = st.checkbox("Show unit circle", value=True)
            show_eigen    = st.checkbox("Show SVD ellipse", value=True)
            grid_lines    = st.slider("Grid density", 5, 31, 13, step=2)

        # ── Error mode banner ─────────────────────────────────────────────

        if st.session_state["error_mode"]:
            st.error(
                "**Error Mode Active** — Matrix `[[1,2],[2,4]]` has det = 0. "
                "This singular matrix collapses 2D space onto a single line. "
                "The inverse does not exist. This is what det = 0 *means*.",
                icon="⚠️",
            )

        # ── Compute Transforms ────────────────────────────────────────────

        M = st.session_state["matrix"]

        # Grid lines: list of (2,2) segments
        grid_segs = get_dense_grid_lines(extent=2.5, n_lines=grid_lines)

        # Unit circle
        circle_pts = get_unit_circle(n_points=300)
        circle_T   = apply_transform(M, circle_pts)

        # Basis vectors: î = [1,0], ĵ = [0,1]
        basis_orig = np.array([[0, 0], [1, 0], [0, 0], [0, 1]], dtype=np.float64)
        basis_T    = apply_transform(M, np.array([[1, 0], [0, 1]], dtype=np.float64))

        # Derived quantities
        det       = compute_determinant(M)
        inv_M, inv_status = compute_inverse(M)
        M_T       = compute_transpose(M)
        U, s, Vt  = decompose_svd(M)
        cond      = condition_number(M)

        # ── Build Plotly Figure ───────────────────────────────────────────

        fig = go.Figure()

        EXTENT = 3.2

        # --- Original grid (blue, dashed, background) ---
        if show_original:
            orig_segs = get_dense_grid_lines(extent=2.5, n_lines=grid_lines)
            for seg in orig_segs:
                fig.add_trace(go.Scatter(
                    x=seg[:, 0], y=seg[:, 1],
                    mode="lines",
                    line=dict(color="rgba(59,130,246,0.20)", width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))

        # --- Transformed grid (green) ---
        for seg in grid_segs:
            seg_T = apply_transform(M, seg)
            fig.add_trace(go.Scatter(
                x=seg_T[:, 0], y=seg_T[:, 1],
                mode="lines",
                line=dict(color="rgba(34,197,94,0.55)", width=1),
                showlegend=False, hoverinfo="skip",
            ))

        # --- Original unit circle (blue, thin) ---
        if show_circle:
            fig.add_trace(go.Scatter(
                x=circle_pts[:, 0], y=circle_pts[:, 1],
                mode="lines", name="Unit circle",
                line=dict(color="rgba(59,130,246,0.4)", width=1.5, dash="dot"),
            ))

            # --- Transformed circle (SVD ellipse) ---
            if show_eigen:
                fig.add_trace(go.Scatter(
                    x=circle_T[:, 0], y=circle_T[:, 1],
                    mode="lines", name="M × Unit circle",
                    line=dict(color="rgba(249,115,22,0.85)", width=2.5),
                    fill="toself",
                    fillcolor="rgba(249,115,22,0.06)",
                ))

        # --- Transformed basis vectors ---
        def _arrow(
            x0: float, y0: float, x1: float, y1: float,
            color: str, label: str
        ) -> go.Scatter:
            return go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines+markers+text",
                line=dict(color=color, width=3),
                marker=dict(symbol="arrow", size=14, color=color,
                            angleref="previous"),
                text=["", label],
                textposition="top right",
                textfont=dict(size=14, color=color),
                name=label,
                showlegend=True,
            )

        # Original basis (faint)
        if show_original:
            fig.add_trace(_arrow(0, 0, 1, 0, "rgba(59,130,246,0.35)", "î (orig)"))
            fig.add_trace(_arrow(0, 0, 0, 1, "rgba(239,68,68,0.35)", "ĵ (orig)"))

        # Transformed basis (bold)
        fig.add_trace(_arrow(0, 0, basis_T[0, 0], basis_T[0, 1],
                             "#3B82F6", "M·î"))
        fig.add_trace(_arrow(0, 0, basis_T[1, 0], basis_T[1, 1],
                             "#EF4444", "M·ĵ"))

        # Origin
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(size=8, color="#475569"),
            showlegend=False, hoverinfo="skip",
        ))

        fig.update_layout(
            xaxis=dict(range=[-EXTENT, EXTENT], zeroline=True,
                       zerolinecolor="#94A3B8", gridcolor="#F1F5F9",
                       scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-EXTENT, EXTENT], zeroline=True,
                       zerolinecolor="#94A3B8", gridcolor="#F1F5F9"),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="#FFFFFF",
            height=580,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0),
            hovermode=False,
        )

        # ── Layout: Figure + Stats ─────────────────────────────────────────

        col_plot, col_stats = st.columns([3, 1])

        with col_plot:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_stats:
            st.subheader("📊 Matrix Properties")

            # Current matrix
            st.markdown("**M**")
            st.code(
                f"[[{M[0,0]:+.3f}, {M[0,1]:+.3f}]\n"
                f" [{M[1,0]:+.3f}, {M[1,1]:+.3f}]]"
            )

            # Determinant
            det_color = "normal" if abs(det) > 0.01 else "inverse"
            st.metric("Determinant", f"{det:+.4f}",
                      delta="area preserved" if abs(abs(det) - 1) < 0.05 else
                            ("singular ⚠️" if abs(det) < 1e-10 else
                             f"area ×{abs(det):.2f}"),
                      delta_color=det_color)

            # Condition number
            cond_display = f"{cond:.1f}" if cond < 1e6 else "∞ (singular)"
            st.metric("Condition number", cond_display,
                      delta="well-conditioned" if cond < 100 else "ill-conditioned ⚠️",
                      delta_color="normal" if cond < 100 else "inverse")

            # Singular values
            st.markdown("**SVD singular values**")
            st.code(f"σ₁ = {s[0]:.4f}\nσ₂ = {s[1]:.4f}")

            # Transpose
            st.markdown("**Transpose M.T**")
            st.code(
                f"[[{M_T[0,0]:+.3f}, {M_T[0,1]:+.3f}]\n"
                f" [{M_T[1,0]:+.3f}, {M_T[1,1]:+.3f}]]"
            )

            # Inverse
            st.markdown("**Inverse M⁻¹**")
            if inv_M is not None:
                st.code(
                    f"[[{inv_M[0,0]:+.3f}, {inv_M[0,1]:+.3f}]\n"
                    f" [{inv_M[1,0]:+.3f}, {inv_M[1,1]:+.3f}]]"
                )
                # Verify: M @ M⁻¹ should be identity
                check = M @ inv_M
                residual = np.max(np.abs(check - np.eye(2)))
                st.caption(f"M @ M⁻¹ residual: {residual:.2e}")
            else:
                st.error(inv_status)

            # Determinant sign interpretation
            st.divider()
            match (det > 0, abs(det) > 1e-10):
                case (True, True):
                    orient = "✅ Orientation preserved"
                case (False, True):
                    orient = "🔄 Orientation flipped"
                case _:
                    orient = "💀 Singular — space collapsed"
            st.markdown(f"**Orientation:** {orient}")

        # ── Explainer ─────────────────────────────────────────────────────

        with st.expander("📖 What am I looking at?", expanded=False):
            st.markdown("""
            **Green grid** = 2D space after transformation M is applied.
            **Orange ellipse** = where M sends the unit circle.
            Its semi-axes equal the **singular values** of M.

            **Blue arrow (M·î)** = where the x-axis unit vector ends up after M.
            **Red arrow (M·ĵ)** = where the y-axis unit vector ends up after M.

            These two arrows are the **columns of M**. Every matrix is just
            "here is where î goes, here is where ĵ goes." Everything else follows.

            **Determinant** = signed area of the parallelogram formed by M·î and M·ĵ.
            Zero determinant = the two output vectors are collinear = space collapses.

            The core computation is one line in `model.py`:
```python
            return points @ M.T   # (N,2) @ (2,2) → (N,2)
```
            """)
    ''')

    # ──────────────────────────────────────────────────────────────────────
    # train.py
    # ──────────────────────────────────────────────────────────────────────
    write(base / "train.py", '''
        """
        train.py — ScratchAI Lesson 01: Matrix Transformer
        ====================================================
        "Training" here means interpolating from the identity matrix to a
        target matrix over N steps and logging the transformation metrics
        at each step — a rigorous stand-in for a gradient descent loop.

        Run:
          python train.py
          python train.py --target rotation --steps 30
          python train.py --target shear --steps 20 --demo

        This teaches students to observe how matrix properties (determinant,
        condition number, singular values) evolve as a transform is applied.
        The same logging discipline applies to weight matrices in real NNs.
        """

        from __future__ import annotations

        import argparse
        import sys
        import time
        from pathlib import Path

        import numpy as np

        from model import (
            IDENTITY,
            PRESETS,
            apply_transform,
            compute_determinant,
            compute_inverse,
            condition_number,
            decompose_svd,
            get_unit_circle,
        )


        def lerp_matrix(
            M_start: np.ndarray,
            M_end: np.ndarray,
            t: float,
        ) -> np.ndarray:
            """
            Linearly interpolate between two matrices.
            t=0 → M_start, t=1 → M_end.
            Component-wise: M(t) = (1-t)*M_start + t*M_end
            """
            return (1.0 - t) * M_start + t * M_end


        def compute_step_metrics(M: np.ndarray, step: int, total: int) -> dict:
            """
            Compute diagnostic metrics for the current transformation matrix.
            These are the same metrics you would watch on a weight matrix
            in a neural network training loop.
            """
            det        = compute_determinant(M)
            _, s, _    = decompose_svd(M)
            cond       = condition_number(M)
            frob_norm  = float(np.linalg.norm(M, "fro"))  # Frobenius norm
            _, inv_status = compute_inverse(M)

            # Apply to unit circle and measure area of output ellipse
            circle = get_unit_circle(200)
            circle_T = apply_transform(M, circle)
            # Area of transformed ellipse ≈ π * σ₁ * σ₂ = π * |det|
            ellipse_area = np.pi * abs(det)

            return {
                "step":        step,
                "total":       total,
                "det":         det,
                "sigma_1":     s[0],
                "sigma_2":     s[1],
                "cond":        cond,
                "frob_norm":   frob_norm,
                "ellipse_area":ellipse_area,
                "invertible":  inv_status == "OK",
            }


        def log_step(m: dict) -> None:
            """Print one training step in structured format."""
            pct   = int(m["step"] / m["total"] * 20)
            bar   = "█" * pct + "░" * (20 - pct)
            inv   = "✓" if m["invertible"] else "✗ singular"

            print(
                f"  [{bar}] step {m['step']:>3}/{m['total']} "
                f"| det={m['det']:+.4f} "
                f"| σ=({m['sigma_1']:.3f},{m['sigma_2']:.3f}) "
                f"| cond={m['cond']:>8.1f} "
                f"| ‖M‖_F={m['frob_norm']:.3f} "
                f"| area={m['ellipse_area']:.3f} "
                f"| inv={inv}"
            )


        def run(
            target_name: str = "Rotation 45°",
            steps: int       = 20,
            demo: bool        = False,
            save_path: Path   = Path("best_weights.npy"),
        ) -> None:
            match target_name.lower():
                case "rotation": target_name = "Rotation 45°"
                case "shear":    target_name = "Shear X"
                case "scale":    target_name = "Scale (2x, 0.5y)"
                case "singular": target_name = "Singular (det=0)"
                case _:          pass  # use as-is, will KeyError cleanly below

            if target_name not in PRESETS:
                print(f"Unknown target '{target_name}'. Valid: {list(PRESETS)}")
                sys.exit(1)

            M_target = PRESETS[target_name]

            print(f"\\n{'━'*70}")
            print(f"  ScratchAI Lesson 01 — Matrix Interpolation Log")
            print(f"  Target : {target_name}")
            print(f"  Steps  : {steps}")
            print(f"{'━'*70}")
            print(
                f"  {'[progress]':>22} "
                f"{'det':>10} "
                f"{'σ₁,σ₂':>14} "
                f"{'cond':>10} "
                f"{'‖M‖_F':>8} "
                f"{'area':>8} "
                f"{'inv':>5}"
            )
            print(f"  {'─'*68}")

            best_cond = float("inf")
            best_M    = IDENTITY.copy()

            t0 = time.perf_counter()

            for step in range(steps + 1):
                t      = step / steps
                M_step = lerp_matrix(IDENTITY, M_target, t)
                metrics = compute_step_metrics(M_step, step, steps)

                log_step(metrics)

                if demo:
                    time.sleep(0.05)  # slow down for live demo mode

                # "Best" = most numerically stable (lowest condition number)
                if metrics["cond"] < best_cond and metrics["invertible"]:
                    best_cond = metrics["cond"]
                    best_M    = M_step.copy()

            elapsed = time.perf_counter() - t0

            print(f"\\n{'━'*70}")
            print(f"  Completed {steps} steps in {elapsed*1000:.1f} ms")
            print(f"  Best condition number : {best_cond:.2f}")
            print(f"  Best matrix saved     : {save_path}")
            print(f"{'━'*70}\\n")

            np.save(save_path, best_M)
            print(f"  Saved best_weights.npy → shape {best_M.shape}")
            print(f"  Load with: M = np.load('{save_path}')\\n")


        def cli() -> None:
            parser = argparse.ArgumentParser(
                description="Matrix Transformer interpolation demo"
            )
            parser.add_argument(
                "--target", default="Rotation 45°",
                help="Preset name: rotation | shear | scale | singular | 'Rotation 45°'"
            )
            parser.add_argument("--steps", type=int, default=20)
            parser.add_argument(
                "--demo", action="store_true",
                help="Slow down output for live demo"
            )
            args = parser.parse_args()
            run(target_name=args.target, steps=args.steps, demo=args.demo)


        if __name__ == "__main__":
            cli()
    ''')

    # ──────────────────────────────────────────────────────────────────────
    # test_model.py
    # ──────────────────────────────────────────────────────────────────────
    write(base / "test_model.py", '''
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
            print(f"\\n{'━'*60}")
            print("  ScratchAI Lesson 01 — Test Suite")
            print(f"{'━'*60}\\n")

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
                    line += f"\\n      └─ {msg}"
                print(line)

            print(f"\\n{'━'*60}")
            print(f"  {passed}/{total} tests passed")
            print(f"{'━'*60}\\n")

            sys.exit(0 if passed == total else 1)


        if __name__ == "__main__":
            main()
    ''')

    # ──────────────────────────────────────────────────────────────────────
    # README.md
    # ──────────────────────────────────────────────────────────────────────
    write(base / "README.md", """
        # ScratchAI — Lesson 01: Matrix Transformer

        **Course:** AI Models From Scratch — Beginner Edition
        **Topic:** Linear Algebra in Code — dot products, matrix multiply,
        transpose, inverse, determinant, SVD

        ---

        ## Run it
```bash
        cd lesson_03
        pip install -r requirements.txt
        streamlit run app.py
```

        Open `http://localhost:8501`. Edit any matrix entry in the sidebar
        and watch the 2D grid transform in real time.

        ---

        ## Demo mode (interpolation log)
```bash
        python train.py --target rotation --steps 30 --demo
        python train.py --target shear    --steps 20
        python train.py --target singular --steps 15
```

        Watch the determinant, condition number, and singular values evolve
        as the matrix interpolates from identity to the target.

        ---

        ## Verify (unit + stress tests)
```bash
        python test_model.py
```

        13 tests: shape contracts, numerical precision,
        linearity property, stress (1000 random batches).

        ---

        ## Break it

        Open `model.py`. In `apply_transform`, change:
```python
        return points @ M.T   # correct
```

        to:
```python
        return M @ points     # wrong — shape (2,2)@(N,2) → crash
```

        Run `python test_model.py`. Read the error. Now you understand
        why the transpose is not optional.

        ---

        ## Extend it (Homework)

        Add homogeneous 3×3 transform support to `model.py`:

        1. Write `to_homogeneous(points)` — appends a column of ones → `(N, 3)`
        2. Write `from_homogeneous(pts_h)` — divides by z, drops z → `(N, 2)`
        3. Modify `apply_transform` to accept 3×3 matrices
        4. Test with a translation matrix `[[1,0,tx],[0,1,ty],[0,0,1]]` —
           translation is *not* achievable with a 2×2 matrix but becomes
           a linear map in homogeneous coordinates

        ---

        ## Cleanup
```bash
        rm -rf __pycache__ *.npy
```

        ---

        ## File Map

        | File            | Purpose                                      |
        |-----------------|----------------------------------------------|
        | `app.py`        | Streamlit UI — live 2D visualization         |
        | `model.py`      | Pure NumPy transforms — the entire lesson    |
        | `train.py`      | Matrix interpolation demo + metric logging   |
        | `test_model.py` | 13 unit + stress tests                       |
        | `requirements.txt` | `numpy>=1.26 streamlit>=1.32 plotly>=5.20` |
    """)

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n✅ lesson_03/ generated successfully.\n")
    print("  Next steps:")
    print("    cd lesson_03")
    print("    pip install -r requirements.txt")
    print("    streamlit run app.py")
    print("    python test_model.py\n")


if __name__ == "__main__":
    main()