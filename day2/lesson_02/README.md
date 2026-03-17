# Lesson 01 — Data as Tensors: Tensor Shape Visualizer
**ScratchAI Beginner Edition · Course 1 of 3**

## Quick Start

```bash
cd lesson_01
pip install -r requirements.txt
streamlit run app.py          # Opens at http://localhost:8501
```

## Commands

| Command | Description |
|---------|-------------|
| `streamlit run app.py` | Launch the interactive visualizer |
| `python train.py` | Run tensor benchmark (all ranks, 3 seeds) |
| `python train.py --epochs 50 --lr 0.01 --demo` | Demo mode benchmark |
| `python train.py --rank 4 --seed 7` | Benchmark a specific rank |
| `python test_model.py` | Run unit tests |
| `python test_stress.py` | Run stress tests (1000 passes) |

## Cleanup

```bash
rm -rf __pycache__ *.npy
```

## What This Lesson Teaches

- Tensor rank (0D–4D) and what each axis represents in practice
- How `.shape`, `.strides`, and `.flags` describe array geometry
- Why `reshape` can silently transpose your data without raising an error
- Vectorized axis reductions vs Python loops (same math, 100× speed gap)
- How to audit an array for NaN, Inf, and value range before training

## File Overview

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard — shape badge, heatmap, axis stats, reshape explorer |
| `model.py` | Pure NumPy tensor analysis — no sklearn, no PyTorch |
| `train.py` | CLI benchmark — logs shape, memory, audit per tensor |
| `test_model.py` | 8 unit tests covering forward pass, stats, reshape, NaN detection |
| `test_stress.py` | 1000-pass stress test — asserts no NaN/Inf, runtime < 5s |

## Homework

Extend `extract_slice` in `model.py` to support 4D tensor exploration:
- Add sidebar sliders for batch index and channel index
- Show `arr[b, :, :, c]` as the primary heatmap
- Add a second Plotly subplot showing the per-channel value histogram

No new libraries required.
