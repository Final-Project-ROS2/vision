# Pixel-to-Real-World Service Documentation

## Overview

The `pixel_to_real_world` service provides accurate conversion from pixel coordinates (u, v) to real-world 3D coordinates (x, y, z) using Intel RealSense depth camera with proper intrinsics and extrinsics.

## Features

- **Aligned Depth Frames**: Aligns depth frames to color frames for accurate depth at pixel locations
- **RealSense SDK Integration**: Uses native RealSense functions for proper coordinate transformation
- **Depth Filtering**: Filters invalid depth readings with configurable min/max range
- **Neighborhood Search**: Falls back to median depth in neighborhood for invalid pixels
- **Intrinsics/Extrinsics**: Uses camera calibration parameters for accurate projection

## Service Interface

**Service Name**: `/pixel_to_real_world`

**Service Type**: `custom_interfaces/srv/PixelToRealWorld`

### Request
```
int32 u     # Pixel column (x-coordinate in image, 0 to width-1)
int32 v     # Pixel row (y-coordinate in image, 0 to height-1)
```

### Response
```
float64 x   # World X coordinate in meters (camera frame)
float64 y   # World Y coordinate in meters (camera frame)
float64 z   # World Z coordinate in meters (camera frame, depth from camera)
```

## Camera Coordinate Frame

The returned coordinates are in the **camera coordinate frame**:
- **X-axis**: Points to the right
- **Y-axis**: Points downward
- **Z-axis**: Points forward (depth from camera)

Origin is at the camera optical center.

## Technical Implementation

### RealSense Configuration

```python
# Depth scale conversion
depth_scale = depth_sensor.get_depth_scale()

# Depth range filtering (meters)
depth_min = 0.11  # 11 cm minimum
depth_max = 1.0   # 1 meter maximum

# Camera intrinsics
depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Extrinsics between streams
depth_to_color_extrin = profile.get_stream(rs.stream.depth).get_extrinsics_to(
    profile.get_stream(rs.stream.color))
color_to_depth_extrin = profile.get_stream(rs.stream.color).get_extrinsics_to(
    profile.get_stream(rs.stream.depth))
```

### Coordinate Transformation Pipeline

1. **Frame Alignment**: Align depth frame to color frame
   ```python
   align = rs.align(rs.stream.color)
   aligned_frames = align.process(frames)
   aligned_depth_frame = aligned_frames.get_depth_frame()
   ```

2. **Depth Reading**: Get depth value at pixel location
   ```python
   depth_value = aligned_depth_frame.get_distance(u, v)
   ```

3. **Depth Validation**: Filter invalid/out-of-range depth
   ```python
   if depth_value < depth_min or depth_value > depth_max:
       # Search neighborhood for valid depth
       depth_value = find_valid_depth_in_neighborhood(u, v)
   ```

4. **Deprojection**: Convert pixel + depth to 3D point
   ```python
   point_3d = rs.rs2_deproject_pixel_to_point(color_intrin, [u, v], depth_value)
   ```

### Alternative Method: Color-to-Depth Projection

For more complex scenarios, you can project color pixels to depth pixels:

```python
color_point = [u, v]
depth_point = rs.rs2_project_color_pixel_to_depth_pixel(
    depth_frame.get_data(),
    depth_scale,
    depth_min,
    depth_max,
    depth_intrin,
    color_intrin,
    depth_to_color_extrin,
    color_to_depth_extrin,
    color_point
)

# Get depth at projected location
depth_value = depth_frame.get_distance(int(depth_point[0]), int(depth_point[1]))

# Deproject using depth intrinsics
point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_point, depth_value)
```

## Setup and Installation

### 1. Build Custom Interfaces
```bash
cd ~/final_project_ws
colcon build --packages-select custom_interfaces
source install/setup.bash
```

### 2. Build Vision Package
```bash
colcon build --packages-select vision --symlink-install
source install/setup.bash
```

### 3. Connect RealSense Camera
Ensure Intel RealSense camera is connected via USB 3.0

### 4. Verify Camera Detection
```bash
realsense-viewer
```

## Usage

### Start the Service

```bash
ros2 run vision pixel_to_real_world_service
```

Expected output:
```
[INFO] [pixel_to_real_world_service]: RealSense pipeline started successfully
[INFO] [pixel_to_real_world_service]: Depth scale: 0.001
[INFO] [pixel_to_real_world_service]: Depth range: 0.11m to 1.0m
[INFO] [pixel_to_real_world_service]: Depth intrinsics: fx=385.45, fy=385.45, ppx=320.00, ppy=240.00
[INFO] [pixel_to_real_world_service]: Color intrinsics: fx=615.79, fy=615.54, ppx=326.42, ppy=238.27
[INFO] [pixel_to_real_world_service]: Retrieved depth-to-color and color-to-depth extrinsics
[INFO] [pixel_to_real_world_service]: Warming up camera...
[INFO] [pixel_to_real_world_service]: Camera ready
[INFO] [pixel_to_real_world_service]: Service /pixel_to_real_world is ready
```

### Call the Service

#### Example 1: Center pixel
```bash
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"
```

Response:
```yaml
x: 0.0234
y: -0.0156
z: 0.6523
```

#### Example 2: Corner pixel
```bash
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 100, v: 100}"
```

#### Example 3: Using Python Client
```python
import rclpy
from rclpy.node import Node
from custom_interfaces.srv import PixelToReal

class PixelToRealClient(Node):
    def __init__(self):
        super().__init__('pixel_to_real_client')
        self.client = self.create_client(PixelToReal, 'pixel_to_real_world')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')
    
    def call_service(self, u, v):
        request = PixelToReal.Request()
        request.u = u
        request.v = v
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(
                f'Pixel ({u}, {v}) -> World ({response.x:.4f}, {response.y:.4f}, {response.z:.4f})')
            return (response.x, response.y, response.z)
        else:
            self.get_logger().error('Service call failed')
            return None

def main():
    rclpy.init()
    client = PixelToRealClient()
    
    # Convert multiple pixels
    pixels = [(320, 240), (400, 300), (200, 150)]
    for u, v in pixels:
        client.call_service(u, v)
    
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting

### Issue: "No RealSense device detected"
**Solution**: 
- Check USB 3.0 connection
- Run `lsusb` to verify camera is detected
- Install librealsense2: `sudo apt install ros-humble-realsense2-camera`

### Issue: "Invalid depth at pixel"
**Solution**:
- Object may be too close (< 11cm) or too far (> 1m)
- Surface may be reflective or transparent
- Service automatically searches neighborhood for valid depth

### Issue: "Service type not available"
**Solution**:
```bash
cd ~/final_project_ws
colcon build --packages-select custom_interfaces
source install/setup.bash
```

### Issue: Depth and color misaligned
**Solution**:
- Service automatically aligns frames using `rs.align(rs.stream.color)`
- Ensure both streams are enabled at same resolution

## Performance Considerations

- **Latency**: ~50-100ms per service call (includes frame capture + processing)
- **Accuracy**: ±2-5mm for objects 0.3-1m from camera
- **Frame Rate**: 30 FPS (depth and color streams)
- **Resolution**: 640x480 (configurable)

## Integration with Other Services

### With Object Detection
```bash
# Terminal 1: Start pixel-to-real service
ros2 run vision pixel_to_real_world_service

# Terminal 2: Start object detection
ros2 run vision simple_sam_detector

# Terminal 3: Get 3D coordinates of detected object centroid
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 350, v: 250}"
```

### With Grasp Planning
```python
# Get object location from detection
bbox_center_u = (bbox['x1'] + bbox['x2']) / 2
bbox_center_v = (bbox['y1'] + bbox['y2']) / 2

# Convert to 3D coordinates
response = pixel_to_real_client.call(u=bbox_center_u, v=bbox_center_v)
object_3d_position = (response.x, response.y, response.z)

# Use for grasp planning
grasp_request.target_position = object_3d_position
```

## Configuration

### Modify Depth Range
Edit the node initialization in `pixel_to_real_world.py`:
```python
self.depth_min = 0.05  # Closer objects
self.depth_max = 2.0   # Further objects
```

### Change Resolution
```python
self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
```

### Adjust Neighborhood Search
```python
def _find_valid_depth_in_neighborhood(self, u, v, window_size=10):  # Larger search area
    # ... search implementation
```

## References

- [Intel RealSense SDK Documentation](https://dev.intelrealsense.com/docs)
- [ROS2 RealSense Wrapper](https://github.com/IntelRealSense/realsense-ros)
- [Depth Camera Calibration Guide](https://www.intelrealsense.com/depth-camera-d435/)

## See Also

- `pixel_to_real.py` - Alternative calibration-based pixel-to-world service
- `find_object_grasp_service_node.py` - Object detection + grasp planning
- `calibration/calibrate.py` - Camera calibration utilities
