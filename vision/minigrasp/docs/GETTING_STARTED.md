# MiniGrasp - Getting Started

## The "No Valid Points" Error - Quick Fix

This error means your workspace bounds don't match your camera setup.

### 3-Step Fix:

#### 1. Run the diagnostic tool
```bash
cd /home/group11/graspnetAPI/minigrasp
python check_camera.py
```

This shows your point cloud and **tells you exactly what bounds to use**.

#### 2. Copy the recommended bounds

The tool will print something like:
```
💡 RECOMMENDED WORKSPACE_BOUNDS for minigrasp/config.py:
{
    'x_min': -0.42,
    'x_max': 0.45,
    'y_min': -0.31,
    'y_max': 0.38,
    'z_min': 0.28,
    'z_max': 0.75,
}
```

Copy these values to `minigrasp/config.py`, replacing the existing `WORKSPACE_BOUNDS`.

#### 3. Test again
```bash
python quick_test.py
```

Should work now!

---

## What Are Workspace Bounds?

Workspace bounds define a 3D box in camera space where grasps are allowed.

**MiniGrasp uses a 3-stage filtering pipeline (recommended for GraspNet):**

### Stage 1: Pass-Through Filter (Workspace Bounds)
Crops to 3D bounding box to focus on object area:

```
       Y (Down)
       |
       |
       |_________ X (Right)
      /
     /
    Z (Forward/Away from camera)
```

- **X**: Left/Right from camera center
- **Y**: Up/Down from camera center  
- **Z**: Distance from camera

### Stage 2: RANSAC Plane Removal
**This is critical!** Removes the table/floor background using RANSAC plane segmentation.

**Why this matters:**
- 90% of points are usually floor/table (wasted computation)
- Floor interferes with collision detection
- Low object-to-background ratio hurts accuracy
- GraspNet works best with **objects only, no background**

### Stage 3: Outlier Removal & Downsampling
- Removes floating noise points
- Creates uniform point density (better for GraspNet)
- Estimates surface normals for quality scoring

**Result:** Clean point cloud with only the graspable object!

If your bounds are:
- **Too small**: "No valid points" error
- **Too large**: Detects background/table instead of object

The diagnostic tool (`check_camera.py`) automatically calculates the right bounds for you!

---

## Minimal Usage Example

Once configured:

```python
from minigrasp import simple_grasp_detector

# One line to detect best grasp
best_grasp = simple_grasp_detector.detect_best_grasp()

if best_grasp:
    pos = best_grasp['position']
    print(f"Grasp at: {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")
```

That's it!

---

## Best Practices

1. **Always run `check_camera.py` first** when setting up on new hardware
2. **Place object 30-80cm from camera** for best results
3. **Use good lighting** - avoid shadows
4. **Start with visualization enabled** to verify detection
5. **Check grasp.score** - aim for > 0.7 confidence

---

## Common Setup Scenarios

### Scenario 1: Table-top picking (camera 50cm above table)
```bash
python check_camera.py  # Shows your exact bounds
# Copy bounds to config.py
python quick_test.py
```

### Scenario 2: Different camera height
Same steps! The diagnostic tool adapts to any setup.

### Scenario 3: Bin picking
Same steps! Just position camera looking into bin, run diagnostic, copy bounds.

---

## Still Having Issues?

1. **Check camera connection**: `realsense-viewer`
2. **Verify object is visible**: Look at visualization from `check_camera.py`
3. **Check permissions**: User must be in `video` group
4. **Update config**: Make sure you copied the bounds correctly

---

## What Makes This "Mini"?

| Feature | MiniGrasp | Full GraspNet |
|---------|-----------|---------------|
| Setup | 3 steps | Complex pipeline |
| Speed | ~2 seconds | ~30 seconds |
| Code | One function | Multiple scripts |
| Output | Best grasp only | All candidates |
| Use case | Quick prototyping | Production |

MiniGrasp = **Minimal setup, maximum simplicity**
