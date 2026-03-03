# MiniGrasp

**Simple, single-shot grasp detection for RealSense depth cameras**

MiniGrasp is a simplified version of GraspNet that captures **one depth image** and returns the **best grasp pose** for detected objects. Perfect for quick prototyping and robotic manipulation projects.

## Features

✅ **One-shot detection** - Capture once, get best grasp  
✅ **Simple API** - Single function call returns grasp pose  
✅ **Full 6D pose** - Position + rotation for robot control  
✅ **Realistic widths** - Calculated from object geometry (0.01-0.05m)  
✅ **RANSAC plane removal** - Removes table/floor background  
✅ **Top-view occlusion handling** - Extrudes 2D surface to 3D volume  
✅ **Grasp refinement** - Aligns grasps to actual point clusters  
✅ **Force estimation** - Suggested gripping force  
✅ **Smart scoring** - Multi-factor quality assessment  
✅ **Configurable** - Easy workspace and gripper setup  
✅ **Visualization** - See detected grasps in 3D  

**Key Advantages:**
- **Automatic background removal** using RANSAC plane segmentation - GraspNet focuses only on graspable objects!
- **Top-view occlusion handling** - Extrudes top surface downward to create synthetic 3D volume from 2D shell
- **Grasp refinement** - Aligns grasp centers to actual point clusters, preventing "floating grasps"  

## Quick Start

### Step 1: Check Camera Setup (IMPORTANT - Run this first!)

```bash
cd minigrasp
python check_camera.py
```

This will:
- Test your camera connection
- Show point cloud statistics
- **Recommend optimal WORKSPACE_BOUNDS** for your setup
- Visualize the point cloud

**Copy the recommended bounds to `minigrasp/config.py`** before proceeding!

### Step 2: Run Quick Test

```bash
python quick_test.py
```

This captures once and detects the best grasp with visualization.

### Step 3: Use in Your Code

```python
from minigrasp import simple_grasp_detector

# Detect best grasp (with visualization)
best_grasp = simple_grasp_detector.detect_best_grasp(visualize=True)

if best_grasp:
    print(f"Position: {best_grasp['position']}")
    print(f"Rotation: {best_grasp['rotation']}")
    print(f"Score: {best_grasp['score']}")
    print(f"Width: {best_grasp['width']}")
```

## Installation

### Prerequisites

- Intel RealSense camera (D435, D455, etc.)
- Python 3.7+
- GraspNetAPI installed

### Dependencies

MiniGrasp uses the existing graspnetAPI environment. Make sure you have:

```bash
# Activate your environment
source /path/to/your/vision_venv/bin/activate

# Verify dependencies (should already be installed)
pip install numpy opencv-python open3d pyrealsense2
```

## API Reference

### `detect_best_grasp(visualize=False)`

Main function to detect best grasp from current camera view.

**Parameters:**
- `visualize` (bool): Show 3D visualization window

**Returns:**
- Dictionary with grasp information:
  ```python
  {
      'position': [x, y, z],              # 3D position (meters)
      'rotation': [[r11, r12, r13],       # 3x3 rotation matrix
                   [r21, r22, r23],
                   [r31, r32, r33]],
      'width': 0.045,                     # Required gripper opening (meters)
      'width_ratio': 0.56,                # % of max gripper capacity
      'score': 0.85,                      # Confidence (0-1)
      'estimated_force': 35.2,            # Required gripping force (N)
      'approach_vector': [x, y, z],       # Movement direction
      'closing_vector': [x, y, z],        # Finger closing direction
      'grasp_speed': 0.3,                 # Recommended speed (0-1)
      'pre_grasp_offset': 0.05,           # Safety distance (meters)
  }
  ```
- Returns `None` if detection fails

**See [VARIABLES.md](VARIABLES.md) for complete variable reference**

### `SimpleGraspDetector` Class

For advanced usage with custom configuration:

```python
from minigrasp.simple_grasp_detector import SimpleGraspDetector
from minigrasp import config

# Customize configuration
config.WORKSPACE_BOUNDS = {
    'x_min': -0.2, 'x_max': 0.2,
    'y_min': -0.2, 'y_max': 0.2,
    'z_min': 0.3, 'z_max': 0.6,
}
config.NUM_GRASP_CANDIDATES = 100
config.MIN_GRASP_SCORE = 0.7

# Create detector with custom config
detector = SimpleGraspDetector()
best_grasp = detector.detect_best_grasp(visualize=True)
```

## Configuration

Edit `minigrasp/config.py` or modify at runtime:

### Camera Settings

```python
MIN_VALID_DEPTH = 0.2  # Minimum depth (meters)
MAX_VALID_DEPTH = 0.8  # Maximum depth (meters)
```

### Workspace Bounds

Define the 3D region where objects can be grasped:

```python
WORKSPACE_BOUNDS = {
    'x_min': -0.3,  # Left
    'x_max': 0.3,   # Right
    'y_min': -0.3,  # Up
    'y_max': 0.3,   # Down
    'z_min': 0.3,   # Near
    'z_max': 0.7,   # Far
}
```

**Coordinate System** (camera frame):
- **X**: Right (+) / Left (-)
- **Y**: Down (+) / Up (-)
- **Z**: Away from camera (+)

### Gripper Parameters

```python
GRIPPER_WIDTH = 0.08   # Max opening (meters)
GRIPPER_DEPTH = 0.05   # Finger depth (meters)
GRIPPER_HEIGHT = 0.02  # Finger height (meters)
```

### Grasp Generation

```python
NUM_GRASP_CANDIDATES = 50  # Number of candidates to generate
MIN_GRASP_SCORE = 0.6      # Minimum quality threshold (0-1)
```

### Approach Filtering

```python
APPROACH_TARGET_VECTOR = [0, 0, -1]  # Preferred approach (downward)
MAX_APPROACH_ANGLE = 60              # Max deviation (degrees)
```

## Preset Configurations

Quick setup for common scenarios:

```python
from minigrasp import config

# Table at 50cm from camera
config.get_table_50cm_config()

# Table at 80cm from camera
config.get_table_80cm_config()

# Bin picking
config.get_bin_picking_config()
```

## Examples

### Example 1: Basic Detection

```python
from minigrasp import simple_grasp_detector

best_grasp = simple_grasp_detector.detect_best_grasp(visualize=True)

if best_grasp:
    print(f"✓ Grasp detected!")
    print(f"  Position: {best_grasp['position']}")
    print(f"  Confidence: {best_grasp['score']:.2%}")
else:
    print("✗ No grasp found")
```

### Example 2: Custom Workspace

```python
from minigrasp import simple_grasp_detector, config

# Configure for your setup
config.WORKSPACE_BOUNDS = {
    'x_min': -0.25, 'x_max': 0.25,
    'y_min': -0.25, 'y_max': 0.25,
    'z_min': 0.2, 'z_max': 0.5,
}

detector = simple_grasp_detector.SimpleGraspDetector()
best_grasp = detector.detect_best_grasp()
```

### Example 3: Robot Integration

```python
from minigrasp import simple_grasp_detector

# Detect grasp
grasp = simple_grasp_detector.detect_best_grasp()

if grasp:
    # Extract pose for robot
    position = grasp['position']        # [x, y, z]
    rotation = grasp['rotation']        # 3x3 matrix
    gripper_width = grasp['width']      # meters
    
    # Send to robot controller
    robot.move_to_pose(position, rotation)
    robot.set_gripper_width(gripper_width)
    robot.grasp()
```

### Example 4: Quality Checking

```python
from minigrasp import simple_grasp_detector

best_grasp = simple_grasp_detector.detect_best_grasp()

if best_grasp:
    if best_grasp['score'] >= 0.8:
        print("✓ High quality grasp - execute")
    elif best_grasp['score'] >= 0.6:
        print("⚠ Medium quality - proceed with caution")
    else:
        print("✗ Low quality - recapture recommended")
```

## Troubleshooting

### "No valid points found" or "Too few points after filtering"

**This is the most common issue!** It means the workspace bounds are too restrictive.

**SOLUTION:**
1. Run the diagnostic tool:
   ```bash
   python minigrasp/check_camera.py
   ```

2. It will show you:
   - Current point cloud range
   - **Recommended WORKSPACE_BOUNDS**

3. Copy the recommended bounds to `minigrasp/config.py`:
   ```python
   WORKSPACE_BOUNDS = {
       'x_min': -0.45,  # Use values from check_camera.py
       'x_max': 0.48,
       'y_min': -0.35,
       'y_max': 0.42,
       'z_min': 0.35,
       'z_max': 0.82,
   }
   ```

4. Run again: `python minigrasp/quick_test.py`

### No grasp detected

**Problem:** `detect_best_grasp()` returns `None`

**Solutions:**
1. Check camera connection
2. Verify object is in workspace bounds
3. Adjust `WORKSPACE_BOUNDS` for your setup
4. Lower `MIN_GRASP_SCORE` threshold
5. Increase `NUM_GRASP_CANDIDATES`

### Too few points after filtering

**Problem:** Warning about insufficient points

**Solutions:**
1. Widen `WORKSPACE_BOUNDS`
2. Check `MIN_VALID_DEPTH` and `MAX_VALID_DEPTH`
3. Verify camera is working: `realsense-viewer`
4. Improve lighting conditions

### Low confidence scores

**Problem:** Grasp scores below 0.7

**Solutions:**
1. Position camera closer to object
2. Ensure good lighting
3. Clean camera lens
4. Verify object is not too reflective
5. Adjust workspace to focus on object

### Camera not found

**Problem:** "Failed to initialize camera"

**Solutions:**
1. Check USB connection
2. Verify camera with: `realsense-viewer`
3. Check udev rules (Linux): `/etc/udev/rules.d/`
4. Try different USB port (USB 3.0 required)

### Grasps look wrong

**Problem:** Detected grasps are misaligned or floating in space

**Solutions:**
1. Enable refinement: `config.ENABLE_GRASP_REFINEMENT = True`
2. Enable top-view extrusion: `config.EXTRUDE_TOP_SURFACE = True`
3. Adjust `config.EXTRUSION_DEPTH` (default 8cm for boxes)
4. Increase `config.MIN_POINTS_FOR_GRASP` for denser clusters
5. Enable `config.CENTER_ON_POINTS = True` for alignment

## Advanced Features

### Top-View Occlusion Handling

**Problem:** When the camera only sees the top of an object (e.g., a box on a table), it creates a 2D shell in 3D space. The network can't determine object thickness, leading to incorrect grasp positioning.

**Solution:** Point cloud extrusion creates synthetic 3D volume from the 2D surface.

```python
# In config.py
EXTRUDE_TOP_SURFACE = True      # Enable extrusion
EXTRUSION_DEPTH = 0.08          # Extrude 8cm downward (for boxes)
EXTRUSION_METHOD = 'uniform'    # Uniform downward extrusion
ASSUMED_OBJECT_HEIGHT = 0.08    # Expected object height
```

How it works:
1. Detects table plane using RANSAC
2. Measures point cloud thickness
3. If thin (< 1cm) = top-view only
4. Extrudes points downward by `EXTRUSION_DEPTH`
5. Creates 5 layers for smooth 3D volume

**When to use:**
- Top-down camera view
- Objects like boxes, books, flat items
- Single-sided point clouds

**When to adjust:**
- `EXTRUSION_DEPTH = 0.05` for shorter objects
- `EXTRUSION_DEPTH = 0.12` for taller objects
- Set `EXTRUDE_TOP_SURFACE = False` for multi-view setups

### Grasp Refinement

**Problem:** Grasps may "float" away from actual points due to sampling/alignment issues.

**Solution:** ICP-like refinement aligns grasp centers to actual point clusters.

```python
# In config.py
ENABLE_GRASP_REFINEMENT = True  # Enable refinement
CENTER_ON_POINTS = True         # Re-center on cluster
MIN_POINTS_FOR_GRASP = 20       # Minimum cluster size
```

How it works:
1. For each grasp, finds nearby points (within 5cm)
2. Removes grasps with too few nearby points
3. Re-centers grasp on cluster centroid
4. Adjusts approach vector to align with local surface normal
5. Only applies reasonable adjustments (< 3cm)

**Benefits:**
- Eliminates "floating grasps" in empty space
- Better alignment with object surface
- Higher success rate in real execution

**Parameters:**
- `MIN_POINTS_FOR_GRASP = 20` - Reject sparse grasps
- `MIN_POINTS_FOR_GRASP = 50` - Stricter clustering
- `CENTER_ON_POINTS = False` - Only filter, don't re-center

**Problem:** Visualization shows unrealistic grasps

**Solutions:**
1. Verify `GRIPPER_WIDTH` matches your gripper
2. Check coordinate frame orientation
3. Adjust `MAX_APPROACH_ANGLE`
4. Review `WORKSPACE_BOUNDS` definition

## Understanding the Output

### Position
The `position` field gives the 3D coordinates (x, y, z) in meters, relative to the camera frame.

### Rotation Matrix
The 3x3 rotation matrix defines the gripper orientation:
- **Column 0** (closing_vector): Direction fingers close
- **Column 1** (binormal): Perpendicular to closing and approach
- **Column 2** (approach_vector): Direction gripper moves toward object

### Score
Confidence value from 0 to 1:
- **0.8-1.0**: High quality, reliable
- **0.6-0.8**: Medium quality, usually acceptable
- **< 0.6**: Low quality, filtered by default

### Width
Required gripper opening distance in meters.

## Coordinate System

MiniGrasp uses the standard camera coordinate system:

```
        Y (Down)
        |
        |
        |_________ X (Right)
       /
      /
     Z (Forward/Away)
```

- **Origin**: Camera optical center
- **X-axis**: Right
- **Y-axis**: Down
- **Z-axis**: Forward (away from camera)

## Architecture

```
1. Camera Capture
   └─> Single depth + color frame
   
2. Point Cloud Generation
   └─> Filter by depth range (0.1-2.0m)
   └─> Convert to 3D points
   
3. Pass-Through Filtering
   └─> Crop to workspace bounds
   
4. RANSAC Plane Removal ⭐ NEW
   └─> Detect dominant plane (table/floor)
   └─> Remove background points
   └─> Keep only object points
   
5. Point Cloud Cleaning
   └─> Remove statistical outliers
   └─> Voxel downsampling (uniform density)
   └─> Estimate surface normals
   
6. Grasp Generation
   └─> Sample candidates around object
   └─> Calculate realistic widths (0.01-0.05m)
   └─> Score based on geometry & stability
   
7. Grasp Filtering
   └─> Score threshold (>0.3)
   └─> Collision detection
   └─> Approach angle check
   
8. Return Best Grasp
   └─> Highest scoring valid grasp
   └─> Width, force, and control parameters
```

**Key Improvement:** RANSAC plane removal eliminates table/floor background, focusing computation on the actual object. This dramatically improves grasp quality for GraspNet-based detection.

## Differences from Full GraspNet

MiniGrasp is **simplified** compared to the full test_depth_camera implementation:

| Feature | MiniGrasp | Full GraspNet |
|---------|-----------|---------------|
| Captures | Single shot | Multi-view |
| Output | Best grasp only | All candidates |
| Network | Random sampling | Trained model |
| Setup | Simple API call | Complex pipeline |
| Speed | ~2-3 seconds | ~10-30 seconds |
| Accuracy | Good for simple objects | Better for complex scenes |

**Use MiniGrasp when:**
- Fast detection needed
- Simple objects
- Prototyping
- Single object in scene

**Use Full GraspNet when:**
- Multiple objects
- Complex shapes
- High accuracy required
- Production deployment

## Tips

1. **Position camera 50-80cm from objects** for best results
2. **Ensure good lighting** - avoid shadows and glare
3. **Start with preset configs** then customize
4. **Use visualization** to verify detection quality
5. **Check scores** - aim for > 0.7 confidence
6. **Calibrate gripper width** to match your hardware
7. **Test workspace bounds** with a ruler

## Contributing

MiniGrasp is part of the GraspNetAPI project. Feel free to:
- Report issues
- Suggest improvements
- Share configurations for different setups

## License

Same as GraspNetAPI (see main repository LICENSE)

## See Also

- Full GraspNet: `test_depth_camera/`
- API Documentation: `docs/`
- Examples: `examples/`
