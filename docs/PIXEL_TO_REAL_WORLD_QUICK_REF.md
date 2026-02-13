# Pixel-to-Real-World Quick Reference

## Quick Start

### 1. Start the Service
```bash
ros2 run vision pixel_to_real_world_service
```

### 2. Call the Service
```bash
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"
```

### 3. Expected Response
```yaml
x: 0.0234
y: -0.0156
z: 0.6523
```

## Service Interface

| Field | Type | Description |
|-------|------|-------------|
| **Request** | | |
| `u` | int32 | Pixel column (0 to 639) |
| `v` | int32 | Pixel row (0 to 479) |
| **Response** | | |
| `x` | float64 | X coordinate in meters |
| `y` | float64 | Y coordinate in meters |
| `z` | float64 | Z coordinate (depth) in meters |

## Coordinate System

```
Camera Frame:
  X → Right
  Y → Down
  Z → Forward (depth)
  
Origin: Camera optical center
```

## Common Pixels

| Location | u | v | Typical Depth |
|----------|---|---|---------------|
| Center | 320 | 240 | 0.5-1.0m |
| Top-left | 0 | 0 | Varies |
| Top-right | 639 | 0 | Varies |
| Bottom-left | 0 | 479 | Varies |
| Bottom-right | 639 | 479 | Varies |

## Configuration Parameters

```python
# Depth range (meters)
depth_min = 0.11  # 11 cm
depth_max = 1.0   # 1 meter

# Resolution
width = 640
height = 480

# Frame rate
fps = 30
```

## Python Client Example

```python
import rclpy
from rclpy.node import Node
from custom_interfaces.srv import PixelToReal

class Client(Node):
    def __init__(self):
        super().__init__('client')
        self.client = self.create_client(PixelToReal, 'pixel_to_real_world')
        self.client.wait_for_service()
    
    def convert(self, u, v):
        request = PixelToReal.Request()
        request.u = u
        request.v = v
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

# Usage
rclpy.init()
client = Client()
response = client.convert(320, 240)
print(f"x={response.x}, y={response.y}, z={response.z}")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No service found | Start service: `ros2 run vision pixel_to_real_world_service` |
| Invalid depth | Object may be too close (<11cm) or too far (>1m) |
| Service not available | Build custom_interfaces: `colcon build --packages-select custom_interfaces` |
| No camera detected | Check USB connection, run `realsense-viewer` |

## Test Scripts

```bash
# Test with bash script
./testsh/test_pixel_to_real_world.sh

# Test with Python client
python3 vision/test_pixel_to_real_world_client.py
```

## Integration Example

```python
# Get object bbox from detection
bbox_center_u = (bbox['x1'] + bbox['x2']) / 2
bbox_center_v = (bbox['y1'] + bbox['y2']) / 2

# Convert to 3D
request = PixelToReal.Request()
request.u = int(bbox_center_u)
request.v = int(bbox_center_v)
response = client.call(request)

# Use for robot control
target_pose.position.x = response.x
target_pose.position.y = response.y
target_pose.position.z = response.z
```

## Performance

- **Latency**: 50-100ms per call
- **Accuracy**: ±2-5mm (0.3-1m range)
- **Frame Rate**: 30 FPS
- **Resolution**: 640x480

## See Also

- Full docs: `docs/PIXEL_TO_REAL_WORLD_SERVICE.md`
- Test client: `vision/test_pixel_to_real_world_client.py`
- Integration: `vision/find_object_grasp_service_node.py`
