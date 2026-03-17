# Browser Output Verification — NumPy Playground

Use this to check that **http://localhost:8501** shows the correct UI and behavior.

---

## 1. Correct layout (what you should see)

### Page title (browser tab)
- **"NumPy Playground | ScratchAI Lesson 01"**

### Sidebar (left)
- Title: **"🔢 NumPy Playground"**
- Caption: **"ScratchAI-Beginner · Lesson 01"**
- **Operation** dropdown with options:
  - Dot Product, Outer Product, Cosine Similarity, Matrix Multiply, Broadcasting Demo, dtype Cast, Vectorization Benchmark
- **Parameters** section (changes with operation):
  - For Dot/Outer/Cosine: **Vector length** slider (2–10), **Vector A** sliders, **Vector B** sliders
  - For Matrix Multiply / Broadcasting: **Rows (m)**, **Cols A / Bias len (n)**, **Cols B (k)**, **Random seed**
  - For dtype Cast: text input for values, **Target dtype** dropdown
  - For Vectorization Benchmark: **Array size** slider
- Two buttons: **"⚠️ Simulate Error"** and **"↺ Reset"**

### Main area (center/right)
- Title: **"NumPy Playground"**
- Caption: *"Every operation below is implemented from scratch in `model.py` using only NumPy. No sklearn. No PyTorch. No magic."*
- Content depends on selected **Operation** (see below).

---

## 2. What each operation does and what to verify

### Dot Product (default)
- **Purpose:** Computes **a · b** = sum of element-wise products; same idea as a single neuron: `weights · inputs + bias`.
- **You should see:**
  - **Left:** Bar chart **"Input Vectors"** — bars for vector **a** (blue) and **b** (green), same length.
  - **Center:** Bar chart **"Element-wise Products"** — orange bars = a[i] × b[i].
  - **Right:** Metric **"a · b"** = one number (e.g. 20.0000); caption with `a.shape` and `b.shape`.
  - Expandable **"Step-by-step breakdown"** with code: `a = ...`, `b = ...`, `elementwise = a * b`, `result = np.sum(elementwise)`.
- **Correctness:** The number **a · b** must equal the sum of the element-wise products (e.g. for a=[1,2,3,4], b=[4,3,2,1] → 1×4+2×3+3×2+4×1 = 20).

### Outer Product
- **Purpose:** Builds matrix **M** with M[i,j] = a[i] × b[j]; shape (len(a), len(b)). Shows broadcasting: `a[:, None] * b[None, :]`.
- **You should see:** Heatmap of matrix **M** with title like "Outer Product: a⊗b → shape (4, 4)", caption with shapes, expandable **"NumPy code"** with the broadcast formula.

### Cosine Similarity
- **Purpose:** Computes (a·b) / (‖a‖‖b‖), in [-1, 1]. Measures angle between vectors; stable for zero vectors.
- **You should see:** Gauge (e.g. -1 to 1) with one value, and metrics for a·b, ‖a‖, ‖b‖; expandable **"Stability note"** with code.

### Matrix Multiply
- **Purpose:** **C = A @ B** with shapes (m,n) @ (n,k) → (m,k). Same as a linear layer.
- **You should see:** Three heatmaps for **A**, **B**, **C = A @ B**; success message like "✓ (3, 4) @ (4, 2) → (3, 2)".

### Broadcasting Demo
- **Purpose:** Adds bias vector **b** to each row of **A**: (m,n) + (n,) → (m,n).
- **You should see:** Heatmaps for **A**, **b** (as bar chart), **A + b**; caption lines showing shape steps.

### dtype Cast
- **Purpose:** Shows precision loss and integer overflow (e.g. int8) when casting.
- **You should see:** Two bar charts (original float64 vs cast), and either a warning (precision/overflow) or success; table of values and differences.

### Vectorization Benchmark
- **Purpose:** Compares Python loop vs NumPy vectorized multiply; shows speedup.
- **You should see:** Metrics for loop time, vectorized time, speedup; bar chart; expandable **"Why the difference?"** text.

### Simulate Error (sidebar button)
- **Purpose:** Demonstrates **Shape Mismatch** (A @ B with wrong shapes) and **int8 Overflow** (e.g. 100+100 in int8).
- **You should see:** Error banner, two tabs with code/messages and hints; **Reset** clears it.

---

## 3. Quick sanity check (default Dot Product)

With **Vector length = 4**, **Vector A** = [1, 2, 3, 4], **Vector B** = [4, 3, 2, 1]:

- **a · b** should be **20.0000** (1×4 + 2×3 + 3×2 + 4×1).
- Element-wise products bar chart should show values **4, 6, 6, 4**.

If that matches, the browser output for Dot Product is correct.

---

## 4. Run a headless check (no browser)

From `lesson_01`:

```bash
.venv/bin/python -c "
from model import dot_product
import numpy as np
a = np.array([1., 2., 3., 4.])
b = np.array([4., 3., 2., 1.])
r = dot_product(a, b)
assert abs(r['result'] - 20.0) < 1e-6
print('Dot product OK: a·b =', r['result'])
"
```

If this prints `Dot product OK: a·b = 20.0`, the backend logic matches the intended behavior.
