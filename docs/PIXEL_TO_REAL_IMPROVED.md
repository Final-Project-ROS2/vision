# Pixel-to-Real-World Coordinate Transformation - Improved Implementation

## Overview
The improved `pixel_to_real_world.py` implementation uses proper pyrealsense2 logic to convert pixel coordinates to real-world 3D coordinates in the robot base frame.

## Coordinate System Specification

### Camera Position
- **Depth Camera Center**: `(0, -0.5442, 0.6711)` meters in base frame
- **Camera Orientation**: Looking straight down at the table

### Reference Frames
- **Base/Table Origin**: `(0, 0, 0)`
- **Floor**: `(0, 0, -0.805)`
- **Table Height**: `z = 0.0`

### Working Area Constraints
- **X range**: `[-0.50, 0.50]` meters (1 meter wide)
- **Y range**: `[-0.70, 0.0]` meters (70 cm depth)
- **Z tolerance**: `±0.05` meters around table height

### Image Specifications
- **Resolution**: 640x480 pixels
- **Pixel range**: `u ∈ [0, 640]`, `v ∈ [0, 480]`

## Key Improvements

### 1. Proper Coordinate Frame Transformation

```python
def camera_to_base_transform(self, x_cam, y_cam, z_cam):
    """Transform from RealSense camera frame to robot base frame."""
    x_base = self.cam_x_base + x_cam
    y_base = self.cam_y_base - z_cam  # Camera Z → Base Y
    z_base = self.cam_z_base - y_cam  # Camera Y → Base Z
    return (x_base, y_base, z_base)
```

**RealSense Camera Frame**:
- X: right
- Y: down
- Z: forward (away from camera)

**Robot Base Frame**:
- X: forward
- Y: left
- Z: up

### 2. Depth Frame Alignment

Uses `rs.align(rs.stream.color)` to align depth frames to color frames, ensuring accurate depth values at pixel locations:

```python
self.align = rs.align(rs.stream.color)
aligned_frames = self.align.process(frames)
self.latest_aligned_depth_frame = aligned_frames.get_depth_frame()
```

### 3. Proper Deprojection with pyrealsense2

Uses `rs2_deproject_pixel_to_point` with color intrinsics for aligned depth:

```python
pixel = [float(u), float(v)]
point_3d_cam = rs.rs2_deproject_pixel_to_point(
    self.color_intrin, 
    pixel, 
    depth_value
)
```

### 4. Working Area Validation

Validates and clamps coordinates to ensure they're within the robot's working area:

```python
def validate_working_area(self, x, y, z):
    """Check if coordinates are within working area."""
    if not (self.x_min <= x <= self.x_max):
        return False
    if not (self.y_min <= y <= self.y_max):
        return False
    if abs(z - self.z_table) > 0.05:
        self.get_logger().warn('Z deviates from table height')
    return True
```

### 5. Robust Depth Handling

- **Depth range**: 0.20m to 1.50m (suitable for camera-to-table distance ~0.67m)
- **Invalid depth handling**: Searches neighborhood (5x5 window) for valid depth values
- **Median filtering**: Uses median of valid neighbors when direct depth is invalid

```python
def _find_valid_depth_in_neighborhood(self, u, v, window_size=5):
    """Find valid depth in neighborhood if pixel depth is invalid."""
    valid_depths = []
    for du in range(-window_size, window_size + 1):
        for dv in range(-window_size, window_size + 1):
            # Check neighbors...
    return float(np.median(valid_depths)) if valid_depths else 0.0
```

## Processing Pipeline

1. **Capture Frames**: Get latest color and depth frames from RealSense
2. **Align Depth**: Align depth frame to color frame resolution
3. **Get Depth Value**: Extract depth at pixel (u, v) with validation
4. **Deproject to 3D**: Convert pixel + depth → 3D point in camera frame
5. **Transform to Base**: Apply coordinate transformation to robot base frame
6. **Validate & Clamp**: Check working area constraints and clamp if needed
7. **Return Result**: Return (x, y, z) in base frame coordinates

## Usage

### Start the Service
```bash
ros2 run vision pixel_to_real_world_service
```

### Call the Service
```bash
# Center of image (320, 240)
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"

# Top-left corner
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 0, v: 0}"

# Bottom-right corner
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 640, v: 480}"
```

### Expected Output
```
x: -0.05    # meters in base frame
y: -0.35    # meters in base frame  
z: 0.0      # meters (table height)
```

## Logging Information

The service provides detailed logging:
- Camera position and working area on startup
- Raw depth values at requested pixels
- Camera frame coordinates (x_cam, y_cam, z_cam)
- Base frame coordinates (x_base, y_base, z_base)
- Working area validation warnings
- Neighborhood search results when needed

## Camera Calibration Notes

The current implementation uses these hardcoded values:
- Camera position: `(0, -0.5442, 0.6711)`
- Camera orientation: Straight down

If the camera position changes, update these constants in the `__init__` method:
```python
self.cam_x_base = 0.0
self.cam_y_base = -0.5442
self.cam_z_base = 0.6711
```

## Dependencies

- `pyrealsense2`: Intel RealSense SDK 2.0
- `numpy`: Numerical operations
- `rclpy`: ROS2 Python client library
- `custom_interfaces`: Custom service definitions

## Advantages Over Previous Implementation

1. ✅ **Proper coordinate transformation** from camera to base frame
2. ✅ **Working area constraints** enforced (x: ±50cm, y: 0 to -70cm)
3. ✅ **Depth validation** with neighborhood search fallback
4. ✅ **Simplified code** - removed complex color-to-depth projection
5. ✅ **Better error handling** with detailed logging
6. ✅ **Clamping to working area** ensures valid robot coordinates
7. ✅ **Table plane assumption** utilized (z ≈ 0)

## Testing Recommendations

1. Test center pixel (320, 240) - should map near camera center projection
2. Test corners to verify working area boundaries
3. Test with objects at known positions to validate accuracy
4. Verify depth values are reasonable (0.5-0.8m for table objects)
5. Check that Y coordinates are negative (toward robot base)
6. Verify Z coordinates are close to 0 (table plane)
