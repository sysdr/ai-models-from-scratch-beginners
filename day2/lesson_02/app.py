"""
app.py — Tensor Shape Visualizer
ScratchAI Beginner · Lesson 01: Data as Tensors

Launch: streamlit run app.py
"""

from __future__ import annotations
import io
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from model import (
    make_demo_tensor,
    compute_shape_stats,
    compute_axis_stats,
    validate_reshape,
    generate_reshape_candidates,
    extract_slice,
    audit_array,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tensor Shape Visualizer · ScratchAI L01",
    page_icon="🔲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Minimal CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .rank-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 8px;
  }
  .error-box {
    background: #FEF2F2;
    border-left: 4px solid #EF4444;
    padding: 12px 16px;
    border-radius: 6px;
    font-family: monospace;
    font-size: 0.88rem;
    color: #991B1B;
  }
  .success-box {
    background: #F0FDF4;
    border-left: 4px solid #22C55E;
    padding: 12px 16px;
    border-radius: 6px;
    font-family: monospace;
    font-size: 0.88rem;
    color: #166534;
  }
  .info-metric {
    background: #EFF6FF;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
  }
</style>
""", unsafe_allow_html=True)

# ─── Session State Defaults ───────────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict = {
        "rank": 2,
        "seed": 42,
        "simulate_error": False,
        "uploaded_arr": None,
        "using_upload": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔲 Tensor Controls")
    st.caption("ScratchAI Beginner · Lesson 01")
    st.divider()

    st.subheader("📊 Demo Tensor")
    rank_sel = st.select_slider(
        "Tensor Rank (0D → 4D)",
        options=[0, 1, 2, 3, 4],
        value=st.session_state["rank"],
        help="0D = scalar, 1D = vector, 2D = matrix, 3D = grayscale batch, 4D = RGB batch",
    )
    st.session_state["rank"] = rank_sel

    seed_sel = st.slider("Random Seed", 1, 200, st.session_state["seed"])
    st.session_state["seed"] = seed_sel

    st.divider()
    st.subheader("📂 Upload CSV")
    uploaded_file = st.file_uploader(
        "Upload a CSV (numeric only)",
        type=["csv"],
        help="First row used as header; all columns must be numeric.",
    )
    if uploaded_file is not None:
        try:
            raw = np.loadtxt(io.StringIO(uploaded_file.read().decode()), delimiter=",", skiprows=1)
            st.session_state["uploaded_arr"] = raw.astype(np.float64)
            st.session_state["using_upload"] = True
            st.success(f"Loaded: shape {raw.shape}")
        except Exception as exc:
            st.error(f"Parse error: {exc}")
            st.session_state["using_upload"] = False
    if st.button("Use Demo Instead"):
        st.session_state["using_upload"] = False
        st.session_state["uploaded_arr"] = None

    st.divider()
    st.subheader("🛠 Actions")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⚠️ Simulate Error", use_container_width=True,
                     help="Trigger a deliberate bad reshape to see what NOT to do"):
            st.session_state["simulate_error"] = True
    with col_b:
        if st.button("↺ Reset", use_container_width=True):
            st.session_state["simulate_error"] = False
            st.session_state["rank"] = 2
            st.session_state["seed"] = 42
            st.session_state["using_upload"] = False
            st.session_state["uploaded_arr"] = None
            st.rerun()

# ─── Load Active Array ────────────────────────────────────────────────────────

if st.session_state["using_upload"] and st.session_state["uploaded_arr"] is not None:
    arr = st.session_state["uploaded_arr"]
    source_label = "📂 Uploaded CSV"
else:
    arr = make_demo_tensor(rank=st.session_state["rank"], seed=st.session_state["seed"])
    source_label = f"🎲 Synthetic {st.session_state['rank']}D Tensor (seed={st.session_state['seed']})"

# ─── Error Simulation Panel ───────────────────────────────────────────────────

if st.session_state["simulate_error"]:
    st.error("## ⚠️  Simulate Error Mode — Bad Reshape")
    st.markdown("""
**What we're doing:** Trying to reshape a 13-element array into shape `(3, 4)`.

`3 × 4 = 12 ≠ 13` — NumPy cannot conserve elements, so it raises:
""")
    st.markdown("""
<div class="error-box">
ValueError: cannot reshape array of size 13 into shape (3,4)<br>
→ Source: np.arange(13).reshape(3, 4)<br>
→ Element count mismatch: 13 (source) ≠ 12 (target = 3×4)<br>
→ Missing 1 element — no valid padding or truncation is applied.
</div>
""", unsafe_allow_html=True)
    st.markdown("""
**The silent version (worse):** Reshaping `(500, 12)` → `(12, 500)` raises NO error
because element count is conserved, but now every row is a feature, not a sample.
Your downstream `.mean(axis=0)` now computes the mean over 500 samples for each of
12 features — **transposed**. The model trains on wrong data. No crash. No warning.

**Fix:** Always verify `arr.shape[0] == n_samples` after every reshape.
""")
    bad = np.arange(13)
    ok, msg = validate_reshape(bad, (3, 4))
    st.code(f"validate_reshape(np.arange(13), (3,4)) → ({ok}, '{msg}')", language="python")
    st.info("Click **↺ Reset** in the sidebar to return to normal mode.")
    st.stop()

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🔲 Tensor Shape Visualizer")
st.caption(f"Source: {source_label}")

profile = compute_shape_stats(arr)
audit = audit_array(arr)

# ─── Top Metrics Row ──────────────────────────────────────────────────────────

RANK_COLORS = {0: "#8B5CF6", 1: "#3B82F6", 2: "#22C55E", 3: "#F97316", 4: "#EF4444"}
rank_color = RANK_COLORS.get(profile.rank, "#475569")
rank_names = {0: "Scalar (0D)", 1: "Vector (1D)", 2: "Matrix (2D)",
              3: "3-D Tensor", 4: "4-D Tensor"}

st.markdown(
    f'<div class="rank-badge" style="background:{rank_color}22; color:{rank_color}; border:1.5px solid {rank_color}88;">'
    f'{rank_names.get(profile.rank, f"{profile.rank}D Tensor")}</div>',
    unsafe_allow_html=True,
)

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Shape", str(profile.shape))
mc2.metric("Rank (ndim)", profile.rank)
mc3.metric("Elements", f"{profile.n_elements:,}")
mc4.metric("Memory", f"{profile.nbytes / 1024:.1f} KB")
mc5.metric("Dtype", profile.dtype)

if audit["has_issues"]:
    st.warning(f"⚠️ Array contains {audit['nan_count']} NaN and {audit['inf_count']} Inf values.")

# ─── Layout: Structural Info + 2D Slice Heatmap ───────────────────────────────

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.subheader("Structural Metadata")

    # Strides table
    if profile.rank >= 1:
        stride_data = {
            "Axis": list(range(profile.rank)),
            "Label": profile.axis_labels,
            "Size": list(profile.shape),
            "Stride (bytes)": list(profile.strides_bytes),
        }
        st.dataframe(stride_data, use_container_width=True, hide_index=True)
    else:
        st.info("0-D scalar — no axes or strides.")

    st.markdown(
        f'<div class="{"success-box" if profile.is_contiguous else "error-box"}">'
        f'Memory layout: {"✓ C-contiguous (row-major)" if profile.is_contiguous else "✗ Non-contiguous — reshape will copy"}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Audit panel
    st.subheader("Array Health")
    hc1, hc2, hc3 = st.columns(3)
    hc1.metric("NaN", audit["nan_count"], delta=None)
    hc2.metric("Inf", audit["inf_count"], delta=None)
    hc3.metric("Zeros", audit["zero_count"], delta=None)
    if np.isfinite(audit["value_min"]):
        st.metric("Value Range", f"[{audit['value_min']:.3f}, {audit['value_max']:.3f}]")

with right_col:
    st.subheader("2-D Slice Heatmap")

    # Controls for slice selection (only meaningful for rank ≥ 3)
    fixed_indices: dict[int, int] = {}
    if profile.rank >= 3:
        st.caption("Fix non-slice axes:")
        fix_cols = st.columns(profile.rank - 2)
        fix_axes = [ax for ax in range(profile.rank) if ax not in (profile.rank - 2, profile.rank - 1)]
        for col_widget, ax in zip(fix_cols, fix_axes):
            max_idx = profile.shape[ax] - 1
            label = profile.axis_labels[ax] if ax < len(profile.axis_labels) else f"axis {ax}"
            fixed_indices[ax] = col_widget.number_input(
                f"{label} index", 0, max_idx, 0, key=f"fix_ax_{ax}"
            )

    slice_axes = (profile.rank - 2, profile.rank - 1) if profile.rank >= 2 else (0, 1)
    s = extract_slice(arr, slice_axes=slice_axes, fixed_indices=fixed_indices)

    fig_heat = go.Figure(
        go.Heatmap(
            z=s.data,
            colorscale="Blues",
            showscale=True,
            hoverongaps=False,
        )
    )
    ax_y_label = profile.axis_labels[slice_axes[0]] if slice_axes[0] < len(profile.axis_labels) else f"axis {slice_axes[0]}"
    ax_x_label = profile.axis_labels[slice_axes[1]] if slice_axes[1] < len(profile.axis_labels) else f"axis {slice_axes[1]}"
    fig_heat.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title=ax_x_label,
        yaxis_title=ax_y_label,
        title=f"arr[{', '.join([':' if ax in slice_axes else str(fixed_indices.get(ax,0)) for ax in range(profile.rank)])}]  shape: {s.data.shape}",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ─── Axis Statistics Charts ───────────────────────────────────────────────────

st.divider()
st.subheader("Axis Statistics (Vectorized Reductions)")
st.caption("`np.mean / std / min / max` with explicit `axis=` — no Python loops.")

if profile.rank >= 2:
    n_axes = min(profile.rank, 3)
    stat_cols = st.columns(n_axes)

    for col_widget, ax in zip(stat_cols, range(n_axes)):
        ax_stats = compute_axis_stats(arr, axis=ax)
        label = profile.axis_labels[ax] if ax < len(profile.axis_labels) else f"axis {ax}"

        mean_flat = ax_stats.mean.ravel()
        std_flat  = ax_stats.std.ravel()
        xs = list(range(len(mean_flat)))

        fig_ax = go.Figure()
        fig_ax.add_trace(go.Bar(
            x=xs, y=mean_flat.tolist(),
            name="mean",
            marker_color="#3B82F6",
            error_y=dict(type="data", array=std_flat.tolist(), visible=True),
        ))
        fig_ax.update_layout(
            title=f"axis={ax} ({label})<br><sup>output shape: {ax_stats.output_shape}</sup>",
            height=240,
            margin=dict(l=0, r=0, t=60, b=0),
            showlegend=False,
            xaxis_title=f"index along axis {1 if ax == 0 else 0}",
            yaxis_title="mean ± std",
        )
        col_widget.plotly_chart(fig_ax, use_container_width=True)
else:
    st.info("Axis stats require rank ≥ 2. Select a higher rank in the sidebar.")

# ─── Reshape Explorer ─────────────────────────────────────────────────────────

st.divider()
st.subheader("Reshape Explorer")
st.caption(f"Total elements: {profile.n_elements:,} — all valid 2-D factorizations shown below.")

candidates = generate_reshape_candidates(arr)

if candidates:
    rows_list = [f"{c.rows} × {c.cols}" for c in candidates]
    is_square  = [c.is_square for c in candidates]
    colors     = ["#22C55E" if sq else "#3B82F6" for sq in is_square]

    fig_reshape = go.Figure(go.Bar(
        x=rows_list,
        y=[c.rows for c in candidates],
        marker_color=colors,
        hovertext=[f"{c.rows}×{c.cols} = {c.rows * c.cols} elements{'  [square]' if c.is_square else ''}" for c in candidates],
        hoverinfo="text+x",
    ))
    fig_reshape.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Reshape (rows × cols)",
        yaxis_title="rows",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_reshape, use_container_width=True)

    # Live reshape validator
    st.markdown("**Try a custom reshape:**")
    rv1, rv2, rv3 = st.columns([2, 2, 3])
    with rv1:
        new_r = st.number_input("rows", min_value=-1, value=int(arr.shape[0]) if profile.rank >= 1 else 1, step=1)
    with rv2:
        new_c = st.number_input("cols", min_value=-1, value=profile.n_elements // max(int(arr.shape[0]), 1) if profile.rank >= 1 else profile.n_elements, step=1)
    with rv3:
        ok, msg = validate_reshape(arr, (new_r, new_c))
        css_class = "success-box" if ok else "error-box"
        icon = "✓" if ok else "✗"
        st.markdown(f'<div class="{css_class}">{icon} {msg}</div>', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption("ScratchAI Beginner · Lesson 01 · Data as Tensors · NumPy only — no PyTorch, no sklearn")
