#!/usr/bin/env python3
"""
test_no_ros.py
==============
Standalone test for the pixel_to_real_world pipeline.
Requires NO ROS — only numpy and the two JSON files.

Tests both stages against every point in calib_data.json and reports
RMSE per height level for:
  • Stage 1  – geometric model only
  • Stage 2  – geometric + residual correction  (needs residual_correction.json)

Run:
    cd d:/yrs4_project/vision/calibration
    python3 test_no_ros.py

Or from anywhere:
    python3 calibration/test_no_ros.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Paths  (everything relative to this file's directory)
# ---------------------------------------------------------------------------
HERE          = os.path.dirname(os.path.abspath(__file__))
CALIB_DATA    = os.path.join(HERE, "calib_data.json")
CORRECTION    = os.path.join(HERE, "residual_correction.json")


# ---------------------------------------------------------------------------
# Geometric helpers  — mirror pixel_to_real_world.py exactly
# ---------------------------------------------------------------------------

def geometric_predict(u, v, height_m, fx, fy, cx, cy, R, t):
    """
    Stage 1: pinhole projection → base frame.
    Z_cam = height_m  (table-plane assumption: object depth ≈ camera height)
    """
    Z_cam = height_m
    x_cam = (u - cx) * Z_cam / fx
    y_cam = (v - cy) * Z_cam / fy
    p_cam = np.array([x_cam, y_cam, Z_cam])
    p_base = R @ p_cam + t
    return float(p_base[0]), float(p_base[1])


def build_feature_vector(x_pred, y_pred, h):
    """
    2nd-order polynomial feature vector.
    F = [x_pred, y_pred, h, x_pred*y_pred, x_pred², y_pred², 1]
    Must match pixel_to_real_world.py::_build_feature_vector exactly.
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


def apply_correction(x_pred, y_pred, h, coef_x, coef_y):
    """Stage 2: polynomial residual correction."""
    F  = build_feature_vector(x_pred, y_pred, h)
    dx = float(F @ coef_x)
    dy = float(F @ coef_y)
    return x_pred + dx, y_pred + dy


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_calib_data():
    if not os.path.exists(CALIB_DATA):
        sys.exit(f"[ERROR] calib_data.json not found at:\n  {CALIB_DATA}")

    with open(CALIB_DATA) as f:
        dataset = json.load(f)

    intr = dataset.get("camera_intrinsics", {})
    fx = float(intr.get("fx", 390.50704956))
    fy = float(intr.get("fy", 390.50704956))
    cx = float(intr.get("cx", 322.57781982))
    cy = float(intr.get("cy", 235.13317871))

    extr = dataset.get("camera_extrinsics", {})
    t = np.array(extr.get("t_base_cam", [-0.109, 0.451, 0.66]), dtype=np.float64)
    R = np.array(extr.get("R_base_cam", [[1,0,0],[0,-1,0],[0,0,-1]]), dtype=np.float64)

    points = [p for p in dataset.get("points", []) if "u" in p and "x_true" in p]
    return fx, fy, cx, cy, R, t, points


def load_correction():
    if not os.path.exists(CORRECTION):
        return None, None, None

    with open(CORRECTION) as f:
        data = json.load(f)

    coef_x = np.array(data["coef_x"], dtype=np.float64)
    coef_y = np.array(data["coef_y"], dtype=np.float64)
    meta   = {
        "rms_before_m": data.get("rms_before_m", float("nan")),
        "rms_after_m":  data.get("rms_after_m",  float("nan")),
        "n_points":     data.get("n_points", "?"),
    }
    return coef_x, coef_y, meta


# ---------------------------------------------------------------------------
# RMSE helpers
# ---------------------------------------------------------------------------

def rmse(errors_m):
    """errors_m: list/array of per-point Euclidean errors in metres."""
    return float(np.sqrt(np.mean(np.square(errors_m))))


def compute_per_height_rmse(records):
    """
    records: list of dicts with keys  height_m, err_geo, err_corr  (metres)
    Returns dict: height → {geo: ..., corr: ...}
    """
    by_height = defaultdict(list)
    for r in records:
        by_height[round(r["height_m"], 4)].append(r)

    result = {}
    for h, pts in sorted(by_height.items()):
        geo_errs  = [p["err_geo"]  for p in pts]
        corr_errs = [p["err_corr"] for p in pts] if pts[0]["err_corr"] is not None else None
        result[h] = {
            "n":    len(pts),
            "geo":  rmse(geo_errs),
            "corr": rmse(corr_errs) if corr_errs is not None else None,
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fx, fy, cx, cy, R, t, points = load_calib_data()
    coef_x, coef_y, corr_meta    = load_correction()
    correction_available          = coef_x is not None

    print()
    print("=" * 70)
    print("  pixel_to_real_world  —  standalone pipeline test  (no ROS)")
    print("=" * 70)
    print(f"  calib_data.json   : {len(points)} points")
    print(f"  fx={fx:.4f}  fy={fy:.4f}  cx={cx:.4f}  cy={cy:.4f}")
    print(f"  t_base_cam        : {t.tolist()}")
    if correction_available:
        m = corr_meta
        print(f"  residual_correction.json loaded  "
              f"(trained on {m['n_points']} pts, "
              f"RMS {m['rms_before_m']*1000:.1f} mm → {m['rms_after_m']*1000:.1f} mm)")
    else:
        print(f"  residual_correction.json NOT FOUND  →  Stage 2 skipped")
        print(f"  Run  calibration/geo_residual_calibrate.py  to generate it.")
    print()

    # ------------------------------------------------------------------
    # Per-point table
    # ------------------------------------------------------------------
    col = "{:>3}  {:>5} {:>5} {:>6}  {:>8} {:>8}  {:>8} {:>8}  {:>8} {:>8}  {:>7}"
    hdr = col.format(
        "#", "u", "v", "h(m)",
        "x_pred", "y_pred",
        "x_true", "y_true",
        "x_final", "y_final",
        "err(mm)"
    )
    print(hdr)
    print("-" * len(hdr))

    records = []
    for i, p in enumerate(points):
        u, v, h  = p["u"], p["v"], p["height_m"]
        xt, yt   = p["x_true"], p["y_true"]

        # Stage 1
        xp, yp   = geometric_predict(u, v, h, fx, fy, cx, cy, R, t)
        err_geo  = float(np.sqrt((xt - xp)**2 + (yt - yp)**2))

        # Stage 2
        if correction_available:
            xf, yf   = apply_correction(xp, yp, h, coef_x, coef_y)
            err_corr = float(np.sqrt((xt - xf)**2 + (yt - yf)**2))
        else:
            xf, yf   = xp, yp
            err_corr = None

        err_display = err_corr if err_corr is not None else err_geo

        print(col.format(
            i + 1, int(u), int(v), f"{h:.2f}",
            f"{xp:.4f}", f"{yp:.4f}",
            f"{xt:.4f}", f"{yt:.4f}",
            f"{xf:.4f}", f"{yf:.4f}",
            f"{err_display*1000:.1f}",
        ))

        records.append({
            "height_m": h,
            "err_geo":  err_geo,
            "err_corr": err_corr,
        })

    # ------------------------------------------------------------------
    # RMSE per height
    # ------------------------------------------------------------------
    per_height = compute_per_height_rmse(records)
    all_geo    = [r["err_geo"]  for r in records]
    all_corr   = [r["err_corr"] for r in records if r["err_corr"] is not None]

    print()
    print("=" * 70)
    print("  RMSE per height")
    print("=" * 70)

    h_col = "{:>8}  {:>5}  {:>14}  {:>14}"
    print(h_col.format("height(m)", "pts", "geo-only (mm)", "corrected (mm)"))
    print("  " + "-" * 50)

    for h, v in per_height.items():
        geo_str  = f"{v['geo']*1000:>12.1f}"
        corr_str = f"{v['corr']*1000:>12.1f}" if v["corr"] is not None else "          n/a"
        print(h_col.format(f"{h:.3f}", v["n"], geo_str, corr_str))

    print("  " + "-" * 50)
    overall_geo  = rmse(all_geo)
    overall_corr = rmse(all_corr) if all_corr else None
    corr_overall_str = (f"{overall_corr*1000:>12.1f}"
                        if overall_corr is not None else "          n/a")
    print(h_col.format("OVERALL", len(records),
                        f"{overall_geo*1000:>12.1f}", corr_overall_str))

    if correction_available and overall_corr is not None:
        factor = overall_geo / max(overall_corr, 1e-9)
        print(f"\n  Improvement factor  :  {factor:.1f}×")
        if overall_corr < 0.010:
            print("  ✓  Correction within 10 mm target — ready for grasping.")
        else:
            print("  ⚠  Correction > 10 mm — consider adding more calibration points.")

    print()


if __name__ == "__main__":
    main()
