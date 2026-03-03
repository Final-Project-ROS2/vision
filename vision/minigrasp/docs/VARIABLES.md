# MiniGrasp Variables Reference

## Complete Output Variables

When you call `detect_best_grasp()`, you get a dictionary with these variables:

### 1. Position & Orientation

```python
best_grasp = detect_best_grasp()

# 3D Position (meters, in camera frame)
position = best_grasp['position']  # [x, y, z]
# Example: [-0.042, 0.015, 0.523]

# Full 3x3 Rotation Matrix
rotation = best_grasp['rotation']  # 3x3 matrix
# Defines gripper orientation in camera frame

# Approach Vector (unit vector)
approach = best_grasp['approach_vector']  # [x, y, z]
# Direction gripper moves to approach object

# Closing Vector (unit vector)
closing = best_grasp['closing_vector']  # [x, y, z]
# Direction gripper fingers close
```

### 2. Gripper Parameters

```python
# Required gripper opening width (meters)
width = best_grasp['width']
# Example: 0.045 (4.5 cm)

# Width as percentage of max gripper capacity
width_ratio = best_grasp['width_ratio']
# Example: 0.56 (using 56% of gripper range)

# Gripper finger height (meters)
height = best_grasp['height']  # Usually 0.02 (2cm)

# Gripper finger depth (meters)
depth = best_grasp['depth']  # Usually 0.02 (2cm)
```

### 3. Force & Quality

```python
# Confidence score (0-1, higher is better)
score = best_grasp['score']
# Example: 0.87 (87% confidence)
# >0.8 = excellent, 0.6-0.8 = good, <0.6 = questionable

# Estimated gripping force needed (Newtons)
force = best_grasp['estimated_force']
# Example: 35.2 N
# Smaller objects need more force, larger need less
```

### 4. Robot Control Parameters

```python
# Pre-grasp offset distance (meters)
offset = best_grasp['pre_grasp_offset']
# Example: 0.05 (5cm)
# Move this distance back along approach vector before closing

# Recommended grasp speed (0-1)
speed = best_grasp['grasp_speed']
# Example: 0.3 for small objects, 0.5 for large
# Slower for delicate/small objects

# Grasp center point (same as position)
center = best_grasp['grasp_center']
```

## Why Each Variable Matters

### `width` - Required Gripper Opening
**Most Important!** This tells your gripper how wide to open before grasping.
- Too wide: Gripper hits object before closing
- Too narrow: Object doesn't fit between fingers
- MiniGrasp calculates this based on actual object geometry

### `width_ratio` - Gripper Utilization
Shows how much of gripper capacity you're using:
- 0.4-0.8 (40-80%) = Ideal range
- <0.3 = Very small object, may slip
- >0.9 = Near max opening, may not close fully

### `estimated_force` - Gripping Force
Helps set gripper force controller:
- Small/narrow objects: Higher force needed (40-50N)
- Large/wide objects: Lower force sufficient (20-30N)
- Prevents crushing delicate items

### `score` - Confidence Level
Quality metric based on:
- Distance from object center (center = better)
- Width utilization (40-80% = optimal)
- Point density (more points = more stable)
- Approach angle (downward = preferred)

### `approach_vector` - Movement Direction
Unit vector showing how to move gripper toward object:
```python
# Move gripper from pre-grasp to grasp position
pre_grasp_pos = position - approach_vector * pre_grasp_offset
# Then move along approach_vector to position
```

### `closing_vector` - Finger Direction
Shows which direction fingers close:
- Perpendicular to approach
- Used to calculate grasp width
- Important for collision checking

### `grasp_speed` - Movement Speed
Recommended speed for grasp execution:
- 0.3 = Slow (small/delicate objects)
- 0.5 = Normal (standard objects)
- Prevents jerky motions that dislodge object

### `pre_grasp_offset` - Safety Distance
How far back to position before approaching:
- Default: 5cm (0.05m)
- Prevents collision during approach
- Gives clear path to final grasp position

## Example Robot Control Sequence

```python
from minigrasp import simple_grasp_detector

# 1. Detect grasp
grasp = simple_grasp_detector.detect_best_grasp()

# 2. Calculate pre-grasp position
pre_grasp = [
    grasp['position'][0] - grasp['approach_vector'][0] * grasp['pre_grasp_offset'],
    grasp['position'][1] - grasp['approach_vector'][1] * grasp['pre_grasp_offset'],
    grasp['position'][2] - grasp['approach_vector'][2] * grasp['pre_grasp_offset']
]

# 3. Execute grasp sequence
robot.open_gripper(width=grasp['width'] + 0.01)  # Open slightly wider
robot.move_to(pre_grasp, grasp['rotation'])      # Move to pre-grasp
robot.move_to(grasp['position'], grasp['rotation'], speed=grasp['grasp_speed'])
robot.close_gripper(force=grasp['estimated_force'])
robot.lift(height=0.1)  # Lift 10cm
```

## Configuration Variables (config.py)

### Gripper Hardware
```python
GRIPPER_WIDTH = 0.08        # Max opening (8cm)
GRIPPER_MIN_WIDTH = 0.01    # Min opening (1cm)
GRIPPER_DEPTH = 0.05        # Finger depth (5cm)
GRIPPER_HEIGHT = 0.02       # Finger height (2cm)
GRIPPER_FORCE = 50.0        # Max force (Newtons)
```

### Workspace Bounds
```python
WORKSPACE_BOUNDS = {
    'x_min': -1.0, 'x_max': 1.0,  # Left/right limits
    'y_min': -1.0, 'y_max': 1.0,  # Up/down limits
    'z_min': 0.2, 'z_max': 1.5,   # Near/far limits
}
```

### Detection Parameters
```python
NUM_GRASP_CANDIDATES = 30   # How many grasps to try
MIN_GRASP_SCORE = 0.3       # Minimum acceptable quality
MAX_APPROACH_ANGLE = 60     # Max deviation from vertical (degrees)
```

### Depth Filtering
```python
MIN_VALID_DEPTH = 0.1       # Minimum depth (10cm)
MAX_VALID_DEPTH = 2.0       # Maximum depth (2m)
```

### Top-View Occlusion Handling (NEW!)
```python
EXTRUDE_TOP_SURFACE = True      # Enable point cloud extrusion
EXTRUSION_DEPTH = 0.08          # Extrude 8cm downward (for boxes)
EXTRUSION_METHOD = 'uniform'    # Uniform downward extrusion
ASSUMED_OBJECT_HEIGHT = 0.08    # Expected object height

# When to adjust:
# - Small objects (cups): EXTRUSION_DEPTH = 0.05
# - Tall objects (bottles): EXTRUSION_DEPTH = 0.12
# - Multi-view setup: EXTRUDE_TOP_SURFACE = False
```

**What it does:**
- Solves the "2D shell problem" where camera only sees top surface
- Extrudes top surface points downward to create synthetic 3D volume
- Helps network determine object thickness and grasp depth

### Grasp Refinement (NEW!)
```python
ENABLE_GRASP_REFINEMENT = True  # Enable ICP-like alignment
CENTER_ON_POINTS = True         # Re-center grasps on clusters
MIN_POINTS_FOR_GRASP = 20       # Minimum cluster size

# When to adjust:
# - Dense clouds: MIN_POINTS_FOR_GRASP = 50
# - Sparse clouds: MIN_POINTS_FOR_GRASP = 10
# - Skip centering: CENTER_ON_POINTS = False
```

**What it does:**
- Aligns grasp centers to actual point clusters
- Removes "floating grasps" in empty space
- Adjusts approach to match local surface normal

## Tips for Best Results

1. **Check `width_ratio`**: Should be 0.4-0.8 for reliable grasps
2. **Check `score`**: Aim for >0.7 for production use
3. **Use `estimated_force`**: Prevents crushing or dropping
4. **Follow `grasp_speed`**: Slower for small/delicate objects
5. **Use `pre_grasp_offset`**: Always approach from safe distance
6. **Enable extrusion**: For top-down camera views
7. **Enable refinement**: For better alignment

## Variable Summary Table

| Variable | Type | Range | Units | Purpose |
|----------|------|-------|-------|---------|
| `position` | list[3] | workspace | meters | Where to grasp |
| `rotation` | list[3][3] | SO(3) | - | Gripper orientation |
| `width` | float | 0.01-0.08 | meters | Gripper opening |
| `width_ratio` | float | 0-1 | - | % of max opening |
| `score` | float | 0-1 | - | Quality/confidence |
| `estimated_force` | float | 20-50 | Newtons | Required force |
| `approach_vector` | list[3] | unit vector | - | Movement direction |
| `closing_vector` | list[3] | unit vector | - | Finger closing dir |
| `grasp_speed` | float | 0-1 | - | Movement speed |
| `pre_grasp_offset` | float | 0.05 | meters | Safety distance |

## Common Questions

**Q: What width should I set my gripper to?**
A: Use `best_grasp['width']` directly, or add 0.5-1cm margin for safety.

**Q: How do I know if the grasp is reliable?**
A: Check `score` > 0.7 and `width_ratio` between 0.4-0.8.

**Q: What force should I use?**
A: Use `estimated_force` as starting point, tune based on object weight.

**Q: Can I ignore some variables?**
A: Minimum needed: `position`, `rotation`, `width`. Others improve reliability.

**Q: How do I convert rotation matrix to euler angles?**
```python
import scipy.spatial.transform as transform
rotation_matrix = best_grasp['rotation']
euler = transform.Rotation.from_matrix(rotation_matrix).as_euler('xyz')
```
