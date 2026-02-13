# Camera Calibration Quick Reference

## Quick Start

```bash
# Terminal 1: Launch camera
ros2 launch ur_yt_sim final_project.launch.py real_camera:=true

# Terminal 2: Run calibration
cd ~/final_project_ws/src/vision/calibration
python3 calibrate.py
```

## Distortion Models (in order of preference)

| Model | When to Use | Error Threshold | Flags |
|-------|-------------|-----------------|-------|
| **k1_only** | Always start here | If error > 0.5px, upgrade | `CALIB_FIX_K2 \| CALIB_FIX_K3` |
| **k1_k2** | k1_only error > 0.5px | If error > 0.5px, upgrade | `CALIB_FIX_K3` |
| **k1_k2_k3** | k1_k2 error > 0.5px + good images | Last resort | None |

## Reprojection Error Guide

| Error (pixels) | Quality | Action |
|----------------|---------|--------|
| < 0.3 | Excellent ✓ | Use this calibration |
| 0.3 - 0.5 | Good ✓ | Use this calibration |
| 0.5 - 1.0 | Acceptable ⚠ | Consider recalibrating |
| > 1.0 | Poor ✗ | Recalibrate or upgrade model |

## Alpha Parameter

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.0 | Crop all black pixels | Maximize useful area |
| 0.5 | Balance | Good compromise |
| 1.0 | Keep all pixels (default) | Preserve all data |

## Capture Tips (Priority Order)

1. ✓ **15-20 sharp, clear images** (better than 40 blurry)
2. ✓ **Cover all 4 corners** of image with chessboard
3. ✓ **Extreme tilts** (rotate X and Y axes)
4. ✓ **Vary depth** (closer and farther)
5. ✓ **Sharp focus** (no motion blur)
6. ✓ **Good lighting** (no shadows/glare)

## Code Snippets

### Load Calibration

```python
import json
import numpy as np
import cv2

with open('camera_calibration.json', 'r') as f:
    calib = json.load(f)

K = np.array(calib['camera_matrix'])
dist = np.array(calib['distortion_coefficients'])
```

### Undistort Image

```python
h, w = img.shape[:2]

# Get optimal new camera matrix (do once)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

# Undistort
dst = cv2.undistort(img, K, dist, None, newcameramtx)

# Optional: crop to ROI
x, y, w, h = roi
dst_cropped = dst[y:y+h, x:x+w]
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| High error (>1.0px) | Recapture with better quality images |
| Large k2/k3 values | Use simpler model (k1_only or k1_k2) |
| Black borders | Use alpha=0 or crop to ROI |
| Chessboard not detected | Better lighting, steady hand, clean lens |

## Menu Options

```
1. Capture new images      ← Start here for new calibration
2. Load & calibrate        ← Use existing images
3. Change distortion model ← Switch between k1_only/k1_k2/k1_k2_k3
4. Exit
```

## File Locations

| File | Location |
|------|----------|
| Script | `~/final_project_ws/src/vision/calibration/calibrate.py` |
| Images | `~/final_project_ws/src/vision/calibration/calibration_images/` |
| Results | `~/final_project_ws/src/vision/calibration/camera_calibration.json` |
| Full Guide | `~/final_project_ws/src/vision/docs/RECALIBRATION_GUIDE.md` |

## Progressive Workflow

```
k1_only → error check → if >0.5px → k1_k2 → error check → if >0.5px → k1_k2_k3
   ↓         ↓                         ↓         ↓                       ↓
  DONE    < 0.5px?                   DONE    < 0.5px?                 DONE
          YES→DONE                            YES→DONE
```

## Keyboard Controls (during capture)

- **SPACE**: Capture image (when chessboard detected)
- **ESC**: Exit capture mode

## Example Output

```
============================================================
CALIBRATION RESULTS
============================================================
Distortion model: k1_only
Number of images used: 20
Image size: (480, 640)
Mean reprojection error: 0.245 pixels  ← Target: < 0.5

Camera Matrix (K):
[[626.47   0.00  317.91]
 [  0.00 624.36  248.60]
 [  0.00   0.00    1.00]]

Distortion Coefficients:
  k1=0.018713
  k2=0.000000  (fixed)
  p1=-0.000055
  p2=0.000589
  k3=0.000000  (fixed)
============================================================
```
