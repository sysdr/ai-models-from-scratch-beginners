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

    print(f"\n{'━'*70}")
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

    print(f"\n{'━'*70}")
    print(f"  Completed {steps} steps in {elapsed*1000:.1f} ms")
    print(f"  Best condition number : {best_cond:.2f}")
    print(f"  Best matrix saved     : {save_path}")
    print(f"{'━'*70}\n")

    np.save(save_path, best_M)
    print(f"  Saved best_weights.npy → shape {best_M.shape}")
    print(f"  Load with: M = np.load('{save_path}')\n")


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
    