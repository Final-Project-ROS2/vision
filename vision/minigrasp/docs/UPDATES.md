# MiniGrasp Updates Summary

## Latest Changes (Top-View Occlusion & Refinement)

### ✅ NEW: Top-View Occlusion Handling

**Problem:** Camera only sees top surface of objects, creating 2D shell in 3D space. Network can't determine object thickness, leading to incorrect grasp depth positioning.

**Solution:** Point cloud extrusion creates synthetic 3D volume from top surface.

```python
# New config parameters
EXTRUDE_TOP_SURFACE = True       # Enable extrusion
EXTRUSION_DEPTH = 0.08           # 8cm default (for boxes)
EXTRUSION_METHOD = 'uniform'     # Uniform downward extrusion
ASSUMED_OBJECT_HEIGHT = 0.08     # Expected object height
```

**Implementation:**
- New method: `_extrude_top_surface(pointcloud, plane_equation)`
- Detects thin point clouds (< 1cm thickness) = top-view only
- Extrudes points downward along table plane normal
- Creates 5 layers for smooth 3D volume
- Points: 10,000 → 50,000 (synthetic volume)

**Impact:** GraspNet now sees complete 3D object volume, not just 2D surface shell!

---

### ✅ NEW: Grasp Refinement & Alignment

**Problem:** Grasps "float" away from actual points due to sampling/alignment issues.

**Solution:** ICP-like refinement aligns grasp centers to actual point clusters.

```python
# New config parameters
ENABLE_GRASP_REFINEMENT = True   # Enable refinement
CENTER_ON_POINTS = True          # Re-center on clusters
MIN_POINTS_FOR_GRASP = 20        # Minimum cluster size
```

**Implementation:**
- New method: `_refine_grasp_candidates(grasp_group, pointcloud)`
- Finds nearby points (within 5cm) for each grasp
- Removes grasps with too few nearby points
- Re-centers grasp on cluster centroid
- Adjusts approach vector to align with local surface normal
- Only applies reasonable adjustments (< 3cm)

**Benefits:**
- Eliminates "floating grasps" in empty space
- Better surface alignment
- Higher execution success rate

---

## Previous Changes

### 1. ✅ Gripper Width: 8cm → 5cm
```python
# config.py
GRIPPER_WIDTH = 0.05  # Changed from 0.08
```

**Impact:** All grasps now respect your actual gripper's 5cm maximum opening.

---

### 2. ✅ RANSAC Plane Removal (Critical Improvement!)

**Problem Identified:**
- Point clouds included 90% table/floor background
- GraspNet wasted computation on impossible grasps
- Floor interfered with collision detection
- Low object-to-background ratio hurt accuracy

**Solution Implemented:**
Added automatic RANSAC plane segmentation to remove table/floor:

```python
# New config parameters
RANSAC_DISTANCE_THRESHOLD = 0.01  # 1cm tolerance
RANSAC_NUM_ITERATIONS = 1000      # Robust plane detection
REMOVE_PLANE = True               # Enabled by default
```

**How it works:**
1. **Pass-Through Filter**: Crop to workspace bounds
2. **RANSAC Plane Detection**: Find dominant plane (table/floor)
3. **Plane Removal**: Keep only object points
4. **Outlier Removal**: Clean floating noise
5. **Voxel Downsampling**: Uniform density for GraspNet

**Before vs After:**
```
Before: 150,000 points (135,000 table + 15,000 object)
After:  15,000 points (object only)

Result: 10x faster, much better grasps!
```

---

### 3. ✅ Realistic Width Calculation

Grasps now calculate required width based on actual object geometry:

```python
def _estimate_grasp_width(points, position, closing_direction):
    # Measures actual object width at grasp point
    # Returns realistic width (not just max gripper)
```

**Benefits:**
- Width varies from 1-5cm based on object size
- Better force estimation
- More reliable grasps

---

### 4. ✅ Enhanced Grasp Quality Scoring

New multi-factor scoring system:

```python
def _calculate_grasp_quality():
    score = 0.5  # Base
    
    # Factor 1: Distance from center (closer = better)
    score += 0.2 * center_proximity
    
    # Factor 2: Width utilization (40-80% = optimal)
    score += 0.15 * width_score
    
    # Factor 3: Point density (more points = stable)
    score += 0.15 * density_score
    
    return score  # 0-1
```

---

### 5. ✅ Comprehensive Output Variables

Added important variables for robot control:

```python
best_grasp = {
    'position': [x, y, z],
    'rotation': [[...], [...], [...]],
    'width': 0.045,              # Calculated, not fixed!
    'width_ratio': 0.56,         # % of gripper capacity
    'score': 0.87,               # Multi-factor quality
    'estimated_force': 35.2,     # Required gripping force
    'approach_vector': [...],    # Movement direction
    'closing_vector': [...],     # Finger direction
    'grasp_speed': 0.3,          # Recommended speed
    'pre_grasp_offset': 0.05,    # Safety distance
}
```

See `VARIABLES.md` for complete reference.

---

## Testing the Improvements

### Step 1: Check camera and get workspace bounds
```bash
cd /home/group11/graspnetAPI/minigrasp
python check_camera.py
```

This shows your point cloud and recommends workspace bounds.

### Step 2: Copy recommended bounds to config.py
```python
# minigrasp/config.py
WORKSPACE_BOUNDS = {
    'x_min': -0.45,  # Use values from check_camera.py
    'x_max': 0.42,
    'y_min': -0.31,
    'y_max': 0.38,
    'z_min': 0.28,
    'z_max': 0.75,
}
```

### Step 3: Test grasp detection
```bash
python quick_test.py
```

You should see:
```
Filtering point cloud (removing background)...
  Raw points: 150000
  → Workspace crop: 80000 points
  → RANSAC plane removal...
    Plane: 0.02x + -0.98y + 0.15z + 0.58 = 0
    Plane points: 65000 (81.2%)
  → After plane removal: 15000 points (object only)
  → Outlier removal: 14500 points
✓ Filtering complete: 14500 object points (background removed)

Generating grasp candidates...
  Object center: [0.015, -0.042, 0.523]
  Object size: [0.085, 0.045, 0.120] m
✓ Generated 30 grasp candidates
  Width range: 1.2 - 4.8 cm
  Score range: 0.621 - 0.891

✓ BEST GRASP DETECTED
Position (x,y,z): [0.018, -0.038, 0.545] m
Confidence: 89.1%

Gripper Settings:
  Width: 3.5 cm (70.0% of max)
  Est. Force: 35.2 N
  Grasp Speed: 0.5 (0=slow, 1=fast)
```

---

## Why These Changes Matter

### For GraspNet Compatibility
✅ Removes background → Better feature extraction  
✅ Uniform point density → Consistent scale estimation  
✅ Object-only points → No wasted computation  
✅ Proper collision checking → Physically valid grasps  

### For Robot Control
✅ Realistic widths → Gripper won't collide  
✅ Force estimation → Prevents crushing/dropping  
✅ Speed hints → Smooth execution  
✅ Pre-grasp offset → Safe approach  

### For Reliability
✅ Multi-factor scoring → Better grasp selection  
✅ Width validation → Gripper compatibility check  
✅ Point density check → Stable contact points  
✅ Approach filtering → Reachable poses  

---

## Configuration Reference

### Essential Settings (config.py)

```python
# Gripper (ADJUST TO YOUR HARDWARE!)
GRIPPER_WIDTH = 0.05  # 5cm max opening

# Workspace (GET FROM check_camera.py!)
WORKSPACE_BOUNDS = {
    'x_min': -0.4, 'x_max': 0.4,
    'y_min': -0.4, 'y_max': 0.4,
    'z_min': 0.2, 'z_max': 1.0,
}

# Plane Removal (RECOMMENDED: KEEP ENABLED)
REMOVE_PLANE = True
RANSAC_DISTANCE_THRESHOLD = 0.01

# Detection
NUM_GRASP_CANDIDATES = 30
MIN_GRASP_SCORE = 0.3
MAX_APPROACH_ANGLE = 60
```

---

## Troubleshooting

### "Too few points after plane removal"
**Cause:** RANSAC removed the object (thinking it's the plane)  
**Fix:** 
1. Object may be very flat → Set `REMOVE_PLANE = False`
2. Workspace too tight → Widen bounds
3. Object is actually on/flush with table → Adjust z_min

### "No grasp found"
**Cause:** All grasps filtered out  
**Fix:**
1. Lower `MIN_GRASP_SCORE` to 0.2
2. Increase `NUM_GRASP_CANDIDATES` to 50
3. Increase `MAX_APPROACH_ANGLE` to 75

### Grasps look unrealistic
**Cause:** Width calculation issues  
**Fix:** Check object is visible in visualization from `check_camera.py`

---

## Best Practices

1. ✅ **Always run `check_camera.py` first** on new hardware
2. ✅ **Copy recommended bounds** to config.py
3. ✅ **Keep plane removal enabled** unless object is very flat
4. ✅ **Verify gripper width** matches your hardware (5cm)
5. ✅ **Check grasp quality** - aim for score > 0.7
6. ✅ **Use visualization** to verify plane removal worked
7. ✅ **Test grasp width** - should be 40-80% of max

---

## Files Modified

- `config.py` - Gripper width, RANSAC parameters
- `simple_grasp_detector.py` - Plane removal, width calculation, quality scoring
- `example.py` - Updated to show new variables
- `README.md` - Architecture section updated
- `GETTING_STARTED.md` - Plane removal explanation
- `VARIABLES.md` - Complete variable reference (NEW)

---

## Migration Guide

If you were using old MiniGrasp:

### Old way:
```python
grasp = detect_best_grasp()
width = grasp['width']  # Always 0.08
```

### New way:
```python
grasp = detect_best_grasp()
width = grasp['width']           # 0.01-0.05 (calculated!)
force = grasp['estimated_force'] # NEW
speed = grasp['grasp_speed']     # NEW
```

All existing code still works, but you get more information now!

---

## Next Steps

1. Run `python check_camera.py` with your setup
2. Copy recommended bounds to `config.py`
3. Run `python quick_test.py` to test
4. Check that plane removal is working (see output)
5. Verify grasp widths are realistic (1-5cm range)
6. Integrate with your robot control code

See `VARIABLES.md` for complete API reference!
