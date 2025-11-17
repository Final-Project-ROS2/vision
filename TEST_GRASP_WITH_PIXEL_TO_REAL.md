# Testing GraspNet with Pixel-to-Real Conversion

## Overview
The GraspNet detector now uses pixel coordinates (u, v) for grasp points and converts them to world coordinates (x, y, z) using the `/pixel_to_real` service.

## Changes Made

### 1. Added PixelToReal Service Client
- Created service client for `/pixel_to_real` 
- Added helper method `_convert_pixel_to_world(u, v)` to convert pixel coordinates to world coordinates

### 2. Updated Grasp Detection
All grasp poses now:
- Use pixel coordinates (u, v) as primary location
- Call `/pixel_to_real` service to get world coordinates (x, y, z)
- Ensure non-zero values for:
  - `quality_score`: minimum 0.1 (range 0.1-1.0)
  - `grasp_width`: minimum 0.02m (2cm)
  - `approach_angle`: minimum 30° (non-zero angles: 30°, 45°, 60°, 90°, 120°, 150°)

### 3. Output Format
Each grasp now includes:
```json
{
  "grasp_id": 0,
  "position": {
    "x": 0.5,        // World X coordinate (meters)
    "y": 0.0,        // World Y coordinate (meters)
    "z": 0.8         // World Z coordinate (meters)
  },
  "orientation": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.383,      // Quaternion Z
    "w": 0.924       // Quaternion W
  },
  "quality_score": 0.7,        // Non-zero: 0.1-1.0
  "grasp_width": 0.05,         // Non-zero: minimum 0.02m
  "approach_angle": 45.0,      // Non-zero: degrees
  "pixel_location": [320, 240], // Original (u, v) coordinates
  "depth_value": 0.8           // Depth in meters
}
```

## Setup Instructions

### Terminal 1: Start pixel_to_real service
```bash
cd ~/final_project_ws
source install/setup.bash
ros2 run vision pixel_to_real_service
```

### Terminal 2: Start SAM detector
```bash
cd ~/final_project_ws
source install/setup.bash
ros2 run vision simple_sam_detector
```

### Terminal 3: Start CLIP classifier
```bash
cd ~/final_project_ws
source install/setup.bash
ros2 run vision clip_classifier
```

### Terminal 4: Start GraspNet detector
```bash
cd ~/final_project_ws
source install/setup.bash
ros2 run vision graspnet_detector
```

## Testing

### Test 1: Manual pixel to real conversion
```bash
# Test green box at pixel (320, 240) should give world (0.5, 0.0, 0.8)
ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"
```

Expected output:
```
x: 0.5
y: 0.0
z: 0.8
```

### Test 2: Detect grasps with automatic conversion
```bash
ros2 service call /vision/detect_grasp std_srvs/srv/Trigger
```

The service will:
1. Detect objects using `/vision/detect_objects`
2. For each object, find grasp point in pixel coordinates (u, v)
3. Call `/pixel_to_real` to convert to world coordinates (x, y, z)
4. Return grasp poses with both pixel and world coordinates

### Test 3: Detect grasp in specific bounding box
```bash
# Detect grasp in region around pixel (320, 240)
ros2 service call /vision/detect_grasp_bb custom_interfaces/srv/DetectGraspBBox "{x1: 250, y1: 180, x2: 390, y2: 300}"
```

### Test 4: Run full pipeline
```bash
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
```

This activates automatic grasp detection when SAM publishes detections.

## Validation Points

The system includes validation for these calibration points:

| Object | Pixel (u, v) | World (x, y, z) |
|--------|--------------|-----------------|
| Origin | (320, 500) | (0.0, 0.0, 0.8) |
| Green Box | (320, 240) | (0.5, 0.0, 0.8) |
| Gear | (305, 95) | (0.83, 0.03, 0.8) |
| Drill | (466, 160) | (0.572, -0.241, 0.832) |
| Monkey Wrench | (150, 200) | (0.624, 0.373, 0.807) |

## Example Output

When running `/vision/detect_grasp`:

```
[INFO] Grasp Detection Service Called
[INFO] Calling /vision/detect_objects service...
[INFO] Detected 3 objects, finding grasps...
[INFO]    Pixel (320, 240) -> World (0.500, 0.000, 0.800)m
[INFO]    Pixel (305, 95) -> World (0.830, 0.030, 0.800)m
[INFO]    Pixel (466, 160) -> World (0.572, -0.241, 0.832)m
[INFO] ✓ Grasp Detection Complete!
[INFO]   Total Objects: 3
[INFO]   Objects with Grasps: 3
[INFO]   Total Grasps: 9
```

## Troubleshooting

### Issue: "PixelToReal service not available"
**Solution**: Make sure pixel_to_real_service is running:
```bash
ros2 run vision pixel_to_real_service
```

### Issue: Grasp quality is 0.0
**Solution**: This should not happen anymore. Quality score is now guaranteed to be >= 0.1

### Issue: Grasp width is 0.0
**Solution**: This should not happen anymore. Grasp width is now guaranteed to be >= 0.02m

### Issue: Approach angle is 0.0
**Solution**: This should not happen anymore. Angles are now: 30°, 45°, 60°, 90°, 120°, 150°

## Coordinate System

- **Image coordinates**: 
  - `u`: pixel column (0 = left, increases right)
  - `v`: pixel row (0 = top, increases down)
  - Origin typically at (320, 500) for 640x480 images

- **World coordinates**:
  - `x`: meters (positive = up, away from table)
  - `y`: meters (positive = left)
  - `z`: meters (height above ground, table at ~0.8m)

## Service Call Examples

```bash
# Example 1: Get world coordinates for pixel (320, 240)
ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"

# Example 2: Detect all grasps
ros2 service call /vision/detect_grasp std_srvs/srv/Trigger

# Example 3: Detect grasp in bbox
ros2 service call /vision/detect_grasp_bb custom_interfaces/srv/DetectGraspBBox "{x1: 100, y1: 100, x2: 300, y2: 300}"

# Example 4: Run full pipeline
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
```

## Notes

- The `/pixel_to_real` service must be running before starting graspnet_detector
- All grasp positions are now accurately converted from image to world coordinates
- Quality scores, grasp widths, and approach angles are guaranteed to be non-zero
- Debug visualization shows both pixel and world coordinates
