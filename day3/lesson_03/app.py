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
                f"[[{M[0,0]:+.3f}, {M[0,1]:+.3f}]
"
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
            st.code(f"σ₁ = {s[0]:.4f}
σ₂ = {s[1]:.4f}")

            # Transpose
            st.markdown("**Transpose M.T**")
            st.code(
                f"[[{M_T[0,0]:+.3f}, {M_T[0,1]:+.3f}]
"
                f" [{M_T[1,0]:+.3f}, {M_T[1,1]:+.3f}]]"
            )

            # Inverse
            st.markdown("**Inverse M⁻¹**")
            if inv_M is not None:
                st.code(
                    f"[[{inv_M[0,0]:+.3f}, {inv_M[0,1]:+.3f}]
"
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
    