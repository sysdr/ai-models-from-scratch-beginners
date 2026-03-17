"""
ScratchAI-Beginner | Lesson 01: NumPy Playground
Streamlit app — launch with: streamlit run app.py
"""

from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from model import (
    dot_product, outer_product, cosine_similarity,
    matrix_multiply, broadcasting_demo, dtype_cast_demo,
    vectorization_benchmark, simulate_shape_error, simulate_dtype_overflow,
)

st.set_page_config(
    page_title="NumPy Playground | ScratchAI Lesson 01",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────
if "error_mode" not in st.session_state:
    st.session_state.error_mode = False
if "operation" not in st.session_state:
    st.session_state.operation = "Dot Product"
if "bench_size" not in st.session_state:
    st.session_state.bench_size = 200_000

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔢 NumPy Playground")
    st.caption("ScratchAI-Beginner · Lesson 01")
    st.divider()

    operation = st.selectbox(
        "Operation",
        [
            "Dot Product",
            "Outer Product",
            "Cosine Similarity",
            "Matrix Multiply",
            "Broadcasting Demo",
            "dtype Cast",
            "Vectorization Benchmark",
        ],
        key="operation",
    )

    st.divider()
    st.subheader("Parameters")

    match operation:
        case "Dot Product" | "Outer Product" | "Cosine Similarity":
            vec_len = st.slider("Vector length", 2, 10, 4)
            st.caption("Vector A")
            a_vals = [
                st.slider(f"a[{i}]", -5.0, 5.0, float(i + 1), 0.1)
                for i in range(vec_len)
            ]
            st.caption("Vector B")
            b_vals = [
                st.slider(f"b[{i}]", -5.0, 5.0, float(vec_len - i), 0.1)
                for i in range(vec_len)
            ]

        case "Matrix Multiply" | "Broadcasting Demo":
            m = st.slider("Rows (m)", 2, 6, 3)
            n = st.slider("Cols A / Bias len (n)", 2, 6, 4)
            k = st.slider("Cols B (k) — matmul only", 2, 6, 2)
            rng_seed = st.slider("Random seed", 0, 99, 42)

        case "dtype Cast":
            raw_vals = st.text_input(
                "Values (comma-separated)", "127.9, 200, -1.7, 0.001, 1e6"
            )
            target_dtype = st.selectbox(
                "Target dtype",
                ["int8", "int16", "int32", "int64", "float32", "float64"],
                index=4,
            )

        case "Vectorization Benchmark":
            bench_size = st.select_slider(
                "Array size",
                options=[10_000, 50_000, 100_000, 200_000, 500_000],
                value=200_000,
                key="bench_size",
            )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚠️ Simulate Error", use_container_width=True):
            st.session_state.error_mode = True
    with col2:
        if st.button("↺ Reset", use_container_width=True):
            st.session_state.error_mode = False

# ── Main Panel ────────────────────────────────────────────────────────────
st.title("NumPy Playground")
st.caption(
    "Every operation below is implemented from scratch in `model.py` "
    "using only NumPy. No sklearn. No PyTorch. No magic."
)

# Error mode display
if st.session_state.error_mode:
    st.error("### ⚠️ Simulate Error Mode")
    err_tab1, err_tab2 = st.tabs(["Shape Mismatch", "int8 Overflow"])
    with err_tab1:
        result = simulate_shape_error()
        st.code(
            f"A = np.ones({result['A_shape']})"
            f"B = np.ones({result['B_shape']})"
            f"C = A @ B"
            f"# → {result['error']}: {result['message']}",
            language="python",
        )
        st.warning(result["hint"])
    with err_tab2:
        result = simulate_dtype_overflow()
        col1, col2, col3 = st.columns(3)
        col1.metric("a", str(list(result["a"])))
        col2.metric("b", str(list(result["b"])))
        col3.metric("a + b (int8)", str(list(result["result_int8"])))
        st.info(
            f"Expected (int64): {list(result['expected_int64'])}  "
            f"Got (int8):       {list(result['result_int8'])}  "
            f"Overflow at:      {list(result['overflow_mask'])}"
        )
        st.warning(result["hint"])
    st.stop()

# ── Operation Outputs ─────────────────────────────────────────────────────
match operation:

    case "Dot Product":
        a = np.array(a_vals, dtype=np.float64)
        b = np.array(b_vals, dtype=np.float64)
        res = dot_product(a, b)

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            fig = go.Figure()
            fig.add_bar(name="a", x=[f"[{i}]" for i in range(len(a))], y=a,
                        marker_color="#3B82F6")
            fig.add_bar(name="b", x=[f"[{i}]" for i in range(len(b))], y=b,
                        marker_color="#22C55E")
            fig.update_layout(title="Input Vectors", barmode="group",
                              height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_bar(
                name="a × b",
                x=[f"[{i}]" for i in range(len(a))],
                y=res["elementwise_products"],
                marker_color="#F97316",
            )
            fig2.update_layout(title="Element-wise Products", height=300,
                               margin=dict(t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)
        with col3:
            st.metric("a · b", f"{res['result']:.4f}")
            st.caption(f"a.shape: {a.shape}\nb.shape: {b.shape}")

        with st.expander("Step-by-step breakdown"):
            st.code(
                f"a = {list(a)}"
                f"b = {list(b)}"
                f"elementwise = a * b  # {list(res['elementwise_products'].round(4))}"
                f"result = np.sum(elementwise)  # {res['result']:.4f}",
                language="python",
            )

    case "Outer Product":
        a = np.array(a_vals, dtype=np.float64)
        b = np.array(b_vals, dtype=np.float64)
        res = outer_product(a, b)
        M = res["result"]

        fig = px.imshow(
            M, text_auto=".2f", color_continuous_scale="Blues",
            title=f"Outer Product: a⊗b  →  shape {M.shape}",
            labels=dict(x="b index (j)", y="a index (i)", color="M[i,j]"),
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"M\[i,j\] = a\[i\] × b\[j\]  |  "
            f"a.shape={a.shape}  b.shape={b.shape}  →  M.shape={M.shape}"
        )
        with st.expander("NumPy code"):
            st.code(
                "# Explicit broadcast form used in model.py:"
                "M = a[:, np.newaxis] * b[np.newaxis, :]"
                f"# a[:, np.newaxis].shape = {a[:, np.newaxis].shape}"
                f"# b[np.newaxis, :].shape = {b[np.newaxis, :].shape}"
                f"# M.shape = {M.shape}",
                language="python",
            )

    case "Cosine Similarity":
        a = np.array(a_vals, dtype=np.float64)
        b = np.array(b_vals, dtype=np.float64)
        res = cosine_similarity(a, b)
        sim = res["similarity"]

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sim,
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "#3B82F6"},
                "steps": [
                    {"range": [-1, -0.5], "color": "#FEE2E2"},
                    {"range": [-0.5, 0.5], "color": "#FEF9C3"},
                    {"range": [0.5, 1],   "color": "#DCFCE7"},
                ],
            },
            title={"text": "Cosine Similarity"},
            number={"valueformat": ".4f"},
        ))
        gauge.update_layout(height=300)
        st.plotly_chart(gauge, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("a · b", f"{res['dot']:.4f}")
        c2.metric("‖a‖", f"{res['norm_a']:.4f}")
        c3.metric("‖b‖", f"{res['norm_b']:.4f}")
        with st.expander("Stability note"):
            st.code(
                "eps = 1e-8"
                "norm_a = np.maximum(np.linalg.norm(a), eps)  # clips zero-norm"
                "norm_b = np.maximum(np.linalg.norm(b), eps)"
                "similarity = np.dot(a, b) / (norm_a * norm_b)"
                "# Then clamp to [-1, 1] to handle float rounding:"
                "similarity = np.clip(similarity, -1.0, 1.0)",
                language="python",
            )

    case "Matrix Multiply":
        rng = np.random.default_rng(rng_seed)
        A = rng.standard_normal((m, n)).round(2)
        B = rng.standard_normal((n, k)).round(2)
        try:
            res = matrix_multiply(A.astype(np.float64), B.astype(np.float64))
            C = res["result"]

            col1, col2, col3 = st.columns(3)
            for matrix, title, col in zip([A, B, C], ["A", "B", "C = A @ B"],
                                           [col1, col2, col3]):
                with col:
                    fig = px.imshow(
                        matrix, text_auto=".1f", color_continuous_scale="Blues",
                        title=f"{title}  shape={matrix.shape}",
                    )
                    fig.update_layout(height=280, margin=dict(t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)

            st.success(
                f"✓ ({m}, {n}) @ ({n}, {k}) → ({m}, {k})  "
                f"inner dimension {n} matches."
            )
        except ValueError as e:
            st.error(f"ValueError: {e}")

    case "Broadcasting Demo":
        rng = np.random.default_rng(rng_seed)
        A = rng.standard_normal((m, n)).round(2)
        b = rng.standard_normal(n).round(2)
        try:
            res = broadcasting_demo(A.astype(np.float64), b.astype(np.float64))
            col1, col2, col3 = st.columns(3)
            with col1:
                fig = px.imshow(A, text_auto=".1f",
                                color_continuous_scale="Blues",
                                title=f"A  shape={A.shape}")
                fig.update_layout(height=280)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = go.Figure()
                fig2.add_bar(
                    x=[f"[{i}]" for i in range(len(b))], y=b,
                    marker_color="#22C55E", name="bias b"
                )
                fig2.update_layout(title=f"b  shape={b.shape}", height=280)
                st.plotly_chart(fig2, use_container_width=True)
            with col3:
                fig3 = px.imshow(res["result"], text_auto=".1f",
                                 color_continuous_scale="Oranges",
                                 title=f"A + b  shape={res['result'].shape}")
                fig3.update_layout(height=280)
                st.plotly_chart(fig3, use_container_width=True)

            for step in res["steps"]:
                st.caption(f"→ {step}")
        except ValueError as e:
            st.error(f"ValueError: {e}")

    case "dtype Cast":
        try:
            vals = [float(v.strip()) for v in raw_vals.split(",")]
            res = dtype_cast_demo(vals, target_dtype)
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_bar(name=f"float64 (original)",
                            x=[str(i) for i in range(len(vals))],
                            y=res["original"], marker_color="#3B82F6")
                fig.update_layout(title="Original (float64)", height=300)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = go.Figure()
                fig2.add_bar(name=target_dtype,
                             x=[str(i) for i in range(len(vals))],
                             y=res["casted"].astype(np.float64),
                             marker_color="#F97316")
                fig2.update_layout(title=f"After cast to {target_dtype}",
                                   height=300)
                st.plotly_chart(fig2, use_container_width=True)

            if res["precision_lost"]:
                st.warning(
                    f"⚠️ Precision lost when casting to `{target_dtype}`. "
                    "Original values differ from casted values."
                )
            elif res["overflow_detected"]:
                st.error(
                    f"⛔ Integer overflow detected in `{target_dtype}`. "
                    "Values wrapped silently."
                )
            else:
                st.success(f"✓ Cast to {target_dtype} is lossless.")

            df_data = {
                "index": list(range(len(vals))),
                "float64": [f"{v:.6g}" for v in res["original"]],
                target_dtype: [str(v) for v in res["casted"]],
                "diff": [
                    f"{abs(float(o) - float(c)):.2e}"
                    for o, c in zip(res["original"], res["casted"].astype(np.float64))
                ],
            }
            st.dataframe(df_data, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    case "Vectorization Benchmark":
        st.info(f"Running benchmark on {bench_size:,} elements…")
        res = vectorization_benchmark(bench_size)
        c1, c2, c3 = st.columns(3)
        c1.metric("Loop time", f"{res['loop_time_ms']:.1f} ms")
        c2.metric("Vectorized time", f"{res['vec_time_ms']:.2f} ms")
        c3.metric("Speedup", f"{res['speedup']:.0f}×")
        st.caption(
            f"Both produce identical results: max diff = {res['max_diff']:.2e}  "
            f"({'✓ identical' if res['identical'] else '⚠ differ'})"
        )
        fig = go.Figure()
        fig.add_bar(
            x=["Python loop", "NumPy vectorized"],
            y=[res["loop_time_ms"], res["vec_time_ms"]],
            marker_color=["#EF4444", "#22C55E"],
            text=[f"{res['loop_time_ms']:.1f} ms", f"{res['vec_time_ms']:.2f} ms"],
            textposition="outside",
        )
        fig.update_layout(
            title=f"Time to multiply {bench_size:,} element pairs",
            yaxis_title="Time (ms)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Why the difference?"):
            st.markdown(
                "The **loop** pays Python overhead on every iteration: "
                "bytecode dispatch, object allocation, type checking, GIL lock.  "
                "The **vectorized** call pays this overhead exactly **once**, "
                "then runs a tight C loop over contiguous float64 memory, "
                "potentially using SIMD (AVX2/AVX-512) to process "
                "8–16 elements per CPU cycle.  "
                "Same math. Different execution model."
            )