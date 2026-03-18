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
    