# Camera Recalibration Guide

## Overview

This guide walks you through the improved camera calibration process using progressive distortion model selection and optimal camera matrix computation.

## What Changed

### 1. Progressive Distortion Model Selection

Instead of using all distortion coefficients (k1, k2, k3) by default, we now use a progressive approach:

- **k1_only** (Recommended Start): Only uses k1, fixes k2 and k3 to zero
  - Simplest model, least prone to overfitting
  - Sufficient for most cameras with minimal fisheye distortion
  - Calibration flags: `cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3`

- **k1_k2** (If Needed): Uses k1 and k2, fixes k3 to zero
  - Use when k1_only shows reprojection error > 0.5 pixels
  - Calibration flags: `cv2.CALIB_FIX_K3`

- **k1_k2_k3** (Use with Caution): Full model with all coefficients
  - Only use if k1_k2 still has large error AND you have high-quality calibration images
  - Can overfit with poor images, leading to worse results
  - Calibration flags: None (all parameters free)

### 2. Optimal New Camera Matrix

The undistortion process now properly uses `cv2.getOptimalNewCameraMatrix()` with configurable alpha:

```python
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha, (w,h))
dst = cv2.undistort(img, K, dist, None, newcameramtx)
```

**Alpha Parameter:**
- `alpha=0`: Crop to remove all black pixels (maximize useful area)
- `alpha=1`: Keep all original pixels visible (may have black borders) - DEFAULT
- `alpha=0.5`: Balance between the two

### 3. ROI Cropping

After undistortion, you can crop to the Region of Interest (ROI) to remove black borders:

```python
x, y, w, h = roi
undistorted_cropped = undistorted[y:y+h, x:x+w]
```

## Calibration Process

### Prerequisites

1. Launch the RealSense camera:
   ```bash
   ros2 launch ur_yt_sim final_project.launch.py real_camera:=true
   ```

2. In another terminal, navigate to the calibration directory:
   ```bash
   cd ~/final_project_ws/src/vision/calibration
   ```

### Step 1: Run Calibration Script

```bash
python3 calibrate.py
```

### Step 2: Select Distortion Model

When the menu appears:
```
Options:
1. Capture new calibration images from ROS2 camera
2. Load existing images and calibrate
3. Change distortion model (current: k1_only)
4. Exit
```

Choose option **3** to change the distortion model. Start with **k1_only**.

### Step 3: Capture Calibration Images

**Quality Tips:**
- **15-20 good images** is better than 40 poor images
- **Cover all corners**: Push chessboard into all 4 image corners
- **Extreme tilts**: Rotate board along X and Y axes
- **Vary depth**: Move board closer and farther from camera
- **Sharp focus**: Ensure no motion blur
- **Good lighting**: Avoid shadows and glare

**Capture Process:**
1. Choose option **1** from menu
2. Enter number of images (recommend 20)
3. Hold chessboard in camera view
4. When green overlay appears (chessboard detected), press **SPACE** to capture
5. Move chessboard to new position/angle
6. Repeat until all images captured
7. Press **ESC** to exit early if needed

### Step 4: Run Calibration

After capturing images, the script will ask:
```
Proceed with calibration? (y/n):
```

Type **y** to run calibration.

### Step 5: Evaluate Results

Check the **mean reprojection error**:

- **< 0.3 pixels**: Excellent calibration ✓
- **0.3 - 0.5 pixels**: Good calibration ✓
- **0.5 - 1.0 pixels**: Acceptable, consider recalibrating
- **> 1.0 pixels**: Poor calibration, recalibrate or try next model

If error is too high with **k1_only**, try **k1_k2**. Only use **k1_k2_k3** if absolutely necessary.

### Step 6: Test Undistortion

When prompted:
```
Test undistortion on an image? (y/n):
```

Type **y** and enter alpha value (recommend starting with **1.0**).

Review the undistorted images:
- **Undistorted (Full)**: All pixels preserved (may have black borders)
- **Undistorted (Cropped to ROI)**: Black borders removed

### Step 7: Save Results

The calibration data is automatically saved to `camera_calibration.json`:

```json
{
    "calibration_date": "2026-02-13 10:30:00",
    "distortion_model": "k1_only",
    "num_images": 20,
    "image_size": [480, 640],
    "camera_matrix": [...],
    "distortion_coefficients": [...],
    "mean_reprojection_error": 0.245
}
```

## Progressive Calibration Workflow

```
┌─────────────────────┐
│  Start with k1_only │
└──────────┬──────────┘
           │
           ▼
     ┌─────────────┐
     │ Error < 0.5?│──Yes──► Use k1_only ✓
     └──────┬──────┘
            │ No
            ▼
   ┌─────────────────┐
   │  Try k1_k2      │
   └────────┬────────┘
            │
            ▼
     ┌─────────────┐
     │ Error < 0.5?│──Yes──► Use k1_k2 ✓
     └──────┬──────┘
            │ No
            ▼
   ┌──────────────────────┐
   │ Good images? > 20?   │──No──► Recapture images
   └──────┬───────────────┘
          │ Yes
          ▼
   ┌─────────────────┐
   │  Try k1_k2_k3   │
   └────────┬────────┘
            │
            ▼
     ┌─────────────┐
     │ Error < 0.5?│──Yes──► Use k1_k2_k3 ✓
     └──────┬──────┘
            │ No
            ▼
   ┌──────────────────────┐
   │ Recapture with more  │
   │ attention to quality │
   └──────────────────────┘
```

## Using Calibration Results in Code

### Loading Calibration Data

```python
import json
import numpy as np

with open('camera_calibration.json', 'r') as f:
    calib = json.load(f)

K = np.array(calib['camera_matrix'])
dist = np.array(calib['distortion_coefficients'])
```

### Undistorting Images

```python
import cv2

# Read image
img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# Get optimal new camera matrix
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

# Undistort
dst = cv2.undistort(img, K, dist, None, newcameramtx)

# Optional: crop to ROI
x, y, w, h = roi
dst_cropped = dst[y:y+h, x:x+w]
```

### Using in ROS2 Nodes

Update your node to use the new calibration:

```python
# In your node's __init__
with open('camera_calibration.json', 'r') as f:
    self.calib = json.load(f)

self.K = np.array(self.calib['camera_matrix'])
self.dist = np.array(self.calib['distortion_coefficients'])

# Get optimal camera matrix once
h, w = 480, 640  # Your image size
self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
    self.K, self.dist, (w, h), 1, (w, h)
)

# In your image callback
def image_callback(self, msg):
    img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    
    # Undistort
    img_undistorted = cv2.undistort(img, self.K, self.dist, None, self.newcameramtx)
    
    # Process undistorted image
    # ...
```

## Common Issues and Solutions

### Issue: High reprojection error (> 1.0 pixels)

**Solutions:**
1. Recapture images with better quality
2. Ensure chessboard is in sharp focus
3. Use more varied poses (corners, tilts, depths)
4. Check chessboard pattern dimensions match code (6x7 corners)
5. Try different distortion model

### Issue: Strange distortion coefficients (very large k2 or k3)

**Cause:** Overfitting due to:
- Too few images
- Poor image quality
- Using k1_k2_k3 unnecessarily

**Solutions:**
1. Start with k1_only
2. Capture more high-quality images
3. Only use k1_k2 or k1_k2_k3 if k1_only has high error

### Issue: Black borders after undistortion

**Solutions:**
1. Use alpha=0 for aggressive cropping
2. Crop to ROI: `img[y:y+h, x:x+w]`
3. Accept alpha=1 and keep black borders (they don't affect pixel coordinates)

### Issue: Chessboard not detected during capture

**Solutions:**
1. Improve lighting
2. Hold chessboard steadier (avoid motion blur)
3. Ensure entire chessboard is visible in frame
4. Clean camera lens
5. Check chessboard print quality (clear black/white squares)

## Best Practices

1. **Start Simple**: Always begin with k1_only
2. **Quality First**: 15 excellent images > 40 mediocre images
3. **Cover the Space**: All corners, tilts, depths
4. **Check Error**: Aim for < 0.5 pixels mean reprojection error
5. **Test Undistortion**: Verify results visually
6. **Document**: Note which model and alpha work best
7. **Version Control**: Save calibration JSON with date

## References

- OpenCV Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Understanding Distortion: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- Optimal Camera Matrix: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga7a6c4e032c97f03ba747966e6ad862b1
