"""
test_stress.py — Stress Tests for Tensor Shape Analyzer
ScratchAI Beginner · Lesson 01: Data as Tensors

Run:  python test_stress.py
"""

from __future__ import annotations
import time
import sys
import numpy as np

from model import (
    compute_shape_stats,
    compute_axis_stats,
    extract_slice,
    audit_array,
    generate_reshape_candidates,
)

PASS = "\033[92m  ✓ PASS\033[0m"
FAIL = "\033[91m  ✗ FAIL\033[0m"


def test_no_nan_inf_stress():
    """
    1000 forward passes with random inputs of varying batch sizes.
    Asserts no NaN/Inf in axis-stat output and runtime < 5s total.
    """
    rng = np.random.default_rng(0)
    shapes = [(b, h, w) for b in [1, 4, 16, 32] for h in [8, 16, 32] for w in [8, 16, 32]]
    # Use 1000 random (batch, h, w) combos
    selected = [shapes[i % len(shapes)] for i in range(1000)]

    t0 = time.perf_counter()
    fail_count = 0

    for i, shape in enumerate(selected):
        arr = rng.standard_normal(shape).astype(np.float64)
        # Axis stats — vectorized reduction
        for ax in range(3):
            stats = compute_axis_stats(arr, axis=ax)
            if np.any(np.isnan(stats.mean)) or np.any(np.isinf(stats.mean)):
                print(f"  NaN/Inf in mean at step {i}, axis {ax}, shape {shape}")
                fail_count += 1
        # Slice
        s = extract_slice(arr, slice_axes=(1, 2))
        if np.any(np.isnan(s.data)) or np.any(np.isinf(s.data)):
            print(f"  NaN/Inf in slice at step {i}, shape {shape}")
            fail_count += 1

    elapsed = time.perf_counter() - t0

    ok_nan = fail_count == 0
    ok_time = elapsed < 5.0

    status = PASS if (ok_nan and ok_time) else FAIL
    print(f"{status}  stress_test_1000_passes  |  {elapsed:.2f}s  |  failures={fail_count}")

    if not ok_nan:
        print(f"       NaN/Inf detected in {fail_count} passes.")
    if not ok_time:
        print(f"       Too slow: {elapsed:.2f}s > 5.0s limit.")

    return ok_nan and ok_time


if __name__ == "__main__":
    print("\nScratchAI L01 · Stress Tests\n" + "─" * 50)
    ok = test_no_nan_inf_stress()
    print("─" * 50)
    if not ok:
        sys.exit(1)
    print("All stress tests passed ✓")
