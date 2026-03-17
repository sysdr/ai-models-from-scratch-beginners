# Lesson 01: NumPy Playground
**ScratchAI-Beginner · Course 1 of 3**

An interactive calculator that performs vector/matrix operations and
explains each step with live output. Built with NumPy only — no ML
frameworks.

## Quick Start
```bash
cd lesson_01
pip install -r requirements.txt
streamlit run app.py
```

## File Structure
```
lesson_01/
├── app.py           # Streamlit UI
├── model.py         # NumPy operations from scratch
├── train.py         # CLI demo + weight learning loop
├── test_model.py    # Unit + stress tests
├── requirements.txt
└── README.md
```

## Lifecycle Commands

| Command | What it does |
|---------|--------------|
| `streamlit run app.py` | Launch the interactive playground |
| `python train.py --epochs 50 --lr 0.01 --demo` | Run learning demo with benchmark |
| `python test_model.py` | Run full test suite |
| `rm -rf __pycache__ *.npy` | Clean up generated files |

## Operations Covered

- **Dot Product** — element-wise multiply + sum; the core of every neuron
- **Outer Product** — broadcasting expansion `a[:, None] * b[None, :]`
- **Cosine Similarity** — normalized dot product with zero-norm stability
- **Matrix Multiply** — shape algebra `(m,k) @ (k,n) → (m,n)`
- **Broadcasting Demo** — `(m,n) + (n,)` with step-by-step shape trace
- **dtype Cast** — precision loss and integer overflow, made visible
- **Vectorization Benchmark** — loop vs vectorized, timed side-by-side

## Homework

Implement `batch_matmul(A, B)` in `model.py`:
- Input: `A` shape `(batch, m, n)`, `B` shape `(batch, n, k)`
- Output: `(batch, m, k)` — **no Python loop over batch**
- Verify: compare to an explicit loop with `atol=1e-10`

Hint: `np.matmul` broadcasts over leading batch dimensions.