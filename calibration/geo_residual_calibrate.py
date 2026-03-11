#!/usr/bin/env python3
"""
Geometric Residual Calibration
================================
This script learns a polynomial *correction* on top of the geometric camera
model used in pixel_to_real_world.py.

Pipeline
--------
1. Load calibration dataset   – calib_data.json
   Each point: (u, v, x_true, y_true, height_m)

2. Geometric prediction (same model as the ROS node)
   Z_cam   = height_m          (table-plane assumption: object depth ≈ camera height)
   x_cam   = (u  - cx) * Z_cam / fx
   y_cam   = (v  - cy) * Z_cam / fy
   p_base  = R @ [x_cam, y_cam, Z_cam] + t  →  (x_pred, y_pred)

3. Residuals
   dx = x_true - x_pred
   dy = y_true - y_pred

4. Feature vector (2nd-order polynomial in predicted position + height)
   F = [x_pred, y_pred, h, x_pred·y_pred, x_pred², y_pred², 1]

   • Linear terms capture constant bias and position-dependent tilt.
   • Cross / quadratic terms capture lens distortion and camera-tilt effects
     (errors that flip direction across the image).
   • Height term captures the perspective-scale shift when the arm moves up/down.

5. OLS fit via numpy.linalg.lstsq (no sklearn required)
   coef_x = argmin ||F @ coef_x - dx||²
   coef_y = argmin ||F @ coef_y - dy||²

6. Diagnostics – RMS before/after correction, per-point residual table

7. Save  calibration/residual_correction.json  (read by the ROS node at startup)

Usage
-----
    cd ~/final_project_ws/src/vision/calibration
    python3 geo_residual_calibrate.py [--data calib_data.json] [--plot]

Options
-------
    --data PATH   Path to the calibration JSON file  (default: calib_data.json)
    --plot        Show residual scatter plot (requires matplotlib)
    --no-save     Dry-run: print coefficients but do not write JSON

Workflow
--------
    1. Collect ≥ 10 calibration points at 2+ camera heights.
       Fill in calib_data.json with your actual measured ground-truth positions.
    2. Run this script to check RMS and inspect residuals.
    3. If RMS-after < 10 mm, save and use the correction in the ROS node.
    4. Re-run whenever the camera mount changes.
"""

import argparse
import json
import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers  (mirror exactly what pixel_to_real_world.py does)
# ---------------------------------------------------------------------------

def build_feature_vector(x_pred: float, y_pred: float, h: float) -> np.ndarray:
    """
    2nd-order polynomial feature vector centred on predicted position + height.

    F = [x_pred, y_pred, h, x_pred*y_pred, x_pred^2, y_pred^2, 1]

    The constant bias term (1) captures the remaining mean offset after the
    geometric transform.  Cross/quadratic terms capture spatial non-linearity
    from lens distortion or a small camera tilt.
    """
    return np.array([
        x_pred,
        y_pred,
        h,
        x_pred * y_pred,
        x_pred ** 2,
        y_pred ** 2,
        1.0,
    ], dtype=np.float64)


def geometric_predict(u: float, v: float, height_m: float,
                      fx: float, fy: float, cx: float, cy: float,
                      R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """
    Compute the geometric base-frame prediction for a single pixel.

    This is the **same formula** as pixel_to_real_world.py (METHOD 1).
    Using (u-cx)*Z/fx directly (avoids the subtle unit-vector error that
    arises if you multiply a normalised ray by Z_depth).
    """
    Z_cam = height_m  # table-plane: object depth ≈ camera height above table
    x_cam = (u - cx) * Z_cam / fx
    y_cam = (v - cy) * Z_cam / fy
    p_cam = np.array([x_cam, y_cam, Z_cam])
    p_base = R @ p_cam + t
    return float(p_base[0]), float(p_base[1])


# ---------------------------------------------------------------------------
# OLS regression
# ---------------------------------------------------------------------------

def fit_correction(features: np.ndarray,
                   dx: np.ndarray,
                   dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ordinary least squares via numpy.linalg.lstsq.

    Returns
    -------
    coef_x : ndarray, shape (7,)
    coef_y : ndarray, shape (7,)
    """
    coef_x, _, _, _ = np.linalg.lstsq(features, dx, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(features, dy, rcond=None)
    return coef_x, coef_y


def apply_correction(x_pred: float, y_pred: float, h: float,
                     coef_x: np.ndarray, coef_y: np.ndarray) -> tuple[float, float]:
    F = build_feature_vector(x_pred, y_pred, h)
    return float(F @ coef_x), float(F @ coef_y)


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def calibrate(data_path: str, save: bool = True, plot: bool = False) -> None:

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    if not os.path.exists(data_path):
        sys.exit(f"[ERROR] Calibration data file not found: {data_path}\n"
                 "        Edit calibration/calib_data.json with your measurements.")

    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Camera intrinsics (P-matrix values – matches PinholeCameraModel)
    intr = dataset.get("camera_intrinsics", {})
    fx = float(intr.get("fx", 390.50704956))
    fy = float(intr.get("fy", 390.50704956))
    cx = float(intr.get("cx", 322.57781982))
    cy = float(intr.get("cy", 235.13317871))

    # Camera extrinsics (matches __init__ of PixelToRealNode)
    extr = dataset.get("camera_extrinsics", {})
    t = np.array(extr.get("t_base_cam", [-0.109, 0.451, 0.66]), dtype=np.float64)
    R = np.array(extr.get("R_base_cam", [[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                 dtype=np.float64)

    raw_points = dataset.get("points", [])
    # Filter out entries that only contain metadata keys
    points = [p for p in raw_points if "u" in p and "x_true" in p]

    if len(points) < 4:
        sys.exit(f"[ERROR] Need at least 4 calibration points, found {len(points)}.\n"
                 "        Fill in calib_data.json with your actual measurements.")

    print(f"\n{'='*60}")
    print(f" Geometric Residual Calibration")
    print(f"{'='*60}")
    print(f" Dataset : {data_path}")
    print(f" Points  : {len(points)}")
    print(f" fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    print(f" t_base_cam = {t.tolist()}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 2. Compute geometric predictions and residuals
    # ------------------------------------------------------------------
    us       = np.array([p["u"]       for p in points], dtype=np.float64)
    vs       = np.array([p["v"]       for p in points], dtype=np.float64)
    x_trues  = np.array([p["x_true"]  for p in points], dtype=np.float64)
    y_trues  = np.array([p["y_true"]  for p in points], dtype=np.float64)
    heights  = np.array([p["height_m"] for p in points], dtype=np.float64)

    x_preds = np.zeros(len(points))
    y_preds = np.zeros(len(points))

    print(f"{'#':>3}  {'u':>5} {'v':>5} {'h':>5}  "
          f"{'x_pred':>8} {'y_pred':>8}  "
          f"{'x_true':>8} {'y_true':>8}  "
          f"{'dx':>8} {'dy':>8}  {'err_m':>7}")
    print("-" * 90)

    for i, p in enumerate(points):
        xp, yp = geometric_predict(
            p["u"], p["v"], p["height_m"], fx, fy, cx, cy, R, t)
        x_preds[i] = xp
        y_preds[i] = yp
        dx = p["x_true"] - xp
        dy = p["y_true"] - yp
        err = np.sqrt(dx**2 + dy**2)
        print(f"{i+1:>3}  {int(p['u']):>5} {int(p['v']):>5} {p['height_m']:>5.3f}  "
              f"{xp:>8.4f} {yp:>8.4f}  "
              f"{p['x_true']:>8.4f} {p['y_true']:>8.4f}  "
              f"{dx:>+8.4f} {dy:>+8.4f}  {err*1000:>6.1f}mm")

    dx_vals = x_trues - x_preds
    dy_vals = y_trues - y_preds
    rms_before = float(np.sqrt(np.mean(dx_vals**2 + dy_vals**2)))
    print(f"\n  RMS (geometric only) : {rms_before*1000:.1f} mm")

    # ------------------------------------------------------------------
    # 3. Build feature matrix and fit residual regression
    # ------------------------------------------------------------------
    F = np.vstack([
        build_feature_vector(x_preds[i], y_preds[i], heights[i])
        for i in range(len(points))
    ])  # shape (N, 7)

    coef_x, coef_y = fit_correction(F, dx_vals, dy_vals)

    # ------------------------------------------------------------------
    # 4. Evaluate corrected predictions
    # ------------------------------------------------------------------
    dx_corr = F @ coef_x
    dy_corr = F @ coef_y

    x_finals = x_preds + dx_corr
    y_finals = y_preds + dy_corr

    res_x_after = x_trues - x_finals
    res_y_after = y_trues - y_finals
    rms_after = float(np.sqrt(np.mean(res_x_after**2 + res_y_after**2)))

    print(f"  RMS (after correction): {rms_after*1000:.1f} mm")
    print(f"\n  Improvement factor: {rms_before/max(rms_after, 1e-9):.1f}×\n")

    # Per-point after-correction residuals
    print(f"{'#':>3}  {'x_final':>8} {'y_final':>8}  "
          f"{'res_x':>8} {'res_y':>8}  {'err_after':>9}")
    print("-" * 60)
    for i in range(len(points)):
        err = np.sqrt(res_x_after[i]**2 + res_y_after[i]**2)
        print(f"{i+1:>3}  {x_finals[i]:>8.4f} {y_finals[i]:>8.4f}  "
              f"{res_x_after[i]:>+8.4f} {res_y_after[i]:>+8.4f}  "
              f"{err*1000:>7.1f}mm")

    # ------------------------------------------------------------------
    # 5. Print learned coefficients
    # ------------------------------------------------------------------
    feature_names = ["x_pred", "y_pred", "h", "x_pred*y_pred",
                     "x_pred²", "y_pred²", "bias"]
    print(f"\n  Correction coefficients")
    print(f"  {'feature':<16}  {'coef_x':>12}  {'coef_y':>12}")
    print(f"  {'-'*42}")
    for name, cx_, cy_ in zip(feature_names, coef_x, coef_y):
        print(f"  {name:<16}  {cx_:>+12.6f}  {cy_:>+12.6f}")

    # ------------------------------------------------------------------
    # 6. Optional scatter plot
    # ------------------------------------------------------------------
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, title, before, after in zip(
                    axes, ["X residuals (mm)", "Y residuals (mm)"],
                    [dx_vals * 1000, dy_vals * 1000],
                    [res_x_after * 1000, res_y_after * 1000]):
                idx = np.arange(1, len(points) + 1)
                ax.plot(idx, before, "o-", label="geometric only", alpha=0.8)
                ax.plot(idx, after, "s-", label="after correction", alpha=0.8)
                ax.axhline(0, color="k", linewidth=0.5)
                ax.set_xlabel("Calibration point #")
                ax.set_ylabel("Residual (mm)")
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("  [WARNING] matplotlib not available – skipping plot.")

    # ------------------------------------------------------------------
    # 7. Save correction JSON
    # ------------------------------------------------------------------
    if not save:
        print("\n  --no-save flag set: skipping write.")
        return

    out_dir = os.path.dirname(os.path.abspath(data_path))
    out_path = os.path.join(out_dir, "residual_correction.json")

    correction = {
        "_description": (
            "Polynomial residual correction for the geometric pixel-to-real model. "
            "Features: [x_pred, y_pred, height, x*y, x^2, y^2, bias]. "
            "Generated by geo_residual_calibrate.py."
        ),
        "feature_order": feature_names,
        "coef_x": coef_x.tolist(),
        "coef_y": coef_y.tolist(),
        "rms_before_m": round(rms_before, 6),
        "rms_after_m":  round(rms_after,  6),
        "n_points": len(points),
        "camera_height_default": float(t[2]),
        "camera_intrinsics": {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy
        },
        "camera_extrinsics": {
            "t_base_cam": t.tolist(),
            "R_base_cam": R.tolist()
        }
    }

    with open(out_path, "w") as f:
        json.dump(correction, f, indent=2)

    print(f"\n  Correction saved → {out_path}")
    print(f"  RMS before: {rms_before*1000:.1f} mm  →  after: {rms_after*1000:.1f} mm")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit geometric residual correction from calibration dataset.")
    parser.add_argument(
        "--data", default=None,
        help="Path to calib_data.json (default: same directory as this script)")
    parser.add_argument(
        "--plot", action="store_true",
        help="Show residual scatter plot (requires matplotlib)")
    parser.add_argument(
        "--no-save", dest="save", action="store_false",
        help="Dry-run: print results without writing residual_correction.json")
    args = parser.parse_args()

    if args.data is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data = os.path.join(script_dir, "calib_data.json")

    calibrate(args.data, save=args.save, plot=args.plot)


if __name__ == "__main__":
    main()
