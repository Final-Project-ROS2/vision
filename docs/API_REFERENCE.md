# Vision Pipeline - API Reference

## ROS2 Services

### 1. Object Detection

**Service:** `/detect_objects`  
**Type:** `vision/srv/DetectObjects`

**Request:**
```
string image_path          # Path to image file
sensor_msgs/Image image    # OR raw image data
---
```

**Response:**
```
bool success
string message
SemanticObject[] objects
  ├─ string label          # Object class (e.g., "cup", "bottle")
  ├─ float32 confidence    # Detection confidence [0-1]
  ├─ geometry_msgs/Pose pose
  └─ geometry_msgs/Point[] mask_points
```

**Example:**
```bash
ros2 service call /detect_objects vision/srv/DetectObjects \
  "{image_path: '/home/user/image.jpg'}"
```

---

### 2. Object Classification

**Service:** `/classify_objects`  
**Type:** `vision/srv/ClassifyRegions`

**Request:**
```
sensor_msgs/Image image
int32[] region_ids         # List of region IDs to classify
BoundingBox[] regions      # Bounding boxes to classify
string[] categories        # Optional: custom categories
---
```

**Response:**
```
bool success
string message
string[] labels            # Classification labels
float32[] confidences      # Confidence scores [0-1]
```

**Example:**
```python
request = ClassifyRegions.Request()
request.image = image_msg
request.categories = ["cup", "bottle", "bowl", "plate"]
response = client.call(request)
```

---

### 3. Grasp Generation

**Service:** `/generate_grasps`  
**Type:** `vision/srv/GenerateGrasps`

**Request:**
```
sensor_msgs/Image rgb_image
sensor_msgs/Image depth_image  # Optional
SemanticObject[] objects
---
```

**Response:**
```
bool success
string message
GraspPose[] grasps
  ├─ geometry_msgs/Pose pose    # 6D grasp pose
  ├─ float32 score              # Grasp quality [0-1]
  └─ string object_label        # Associated object
```

---

### 4. Scene Graph Building

**Service:** `/build_scene_graph`  
**Type:** `vision/srv/BuildSceneGraph`

**Request:**
```
SemanticObject[] objects
---
```

**Response:**
```
bool success
string message
SceneGraph scene_graph
  ├─ SemanticObject[] objects
  └─ SpatialRelation[] relations
       ├─ string subject         # Object 1
       ├─ string relation        # "on", "next_to", "above", etc.
       └─ string object          # Object 2
```

---

## ROS2 Topics

### Published Topics

**`/camera/image_raw`**  
Type: `sensor_msgs/Image`  
Description: RGB camera feed

**`/camera/depth`**  
Type: `sensor_msgs/Image`  
Description: Depth image (RealSense mode only)

**`/detections/visualization`**  
Type: `sensor_msgs/Image`  
Description: Annotated detection results

---

## Python API

### Using the Pipeline from Python

```python
import rclpy
from rclpy.node import Node
from vision.srv import DetectObjects, ClassifyRegions

class VisionClient(Node):
    def __init__(self):
        super().__init__('vision_client')
        
        # Create service clients
        self.detect_client = self.create_client(
            DetectObjects, '/detect_objects')
        self.classify_client = self.create_client(
            ClassifyRegions, '/classify_objects')
    
    def detect_and_classify(self, image_path):
        # Step 1: Detect objects
        detect_req = DetectObjects.Request()
        detect_req.image_path = image_path
        
        detect_future = self.detect_client.call_async(detect_req)
        rclpy.spin_until_future_complete(self, detect_future)
        detect_resp = detect_future.result()
        
        if not detect_resp.success:
            self.get_logger().error(f"Detection failed: {detect_resp.message}")
            return None
        
        # Step 2: Classify detected objects
        classify_req = ClassifyRegions.Request()
        classify_req.image = detect_resp.annotated_image
        classify_req.regions = detect_resp.bounding_boxes
        
        classify_future = self.classify_client.call_async(classify_req)
        rclpy.spin_until_future_complete(self, classify_future)
        classify_resp = classify_future.result()
        
        return {
            'objects': detect_resp.objects,
            'labels': classify_resp.labels,
            'confidences': classify_resp.confidences
        }

# Usage
def main():
    rclpy.init()
    client = VisionClient()
    results = client.detect_and_classify('/path/to/image.jpg')
    print(f"Found {len(results['objects'])} objects")
    rclpy.shutdown()
```

---

## Command Line Interface

### Quick Commands

```bash
# Detect objects in image
ros2 service call /detect_objects vision/srv/DetectObjects \
  "{image_path: '/path/to/image.jpg'}"

# List available services
ros2 service list | grep vision

# Check service type
ros2 service type /detect_objects

# Monitor camera topic
ros2 topic echo /camera/image_raw

# View detection rate
ros2 topic hz /detections/visualization
```

---

## Configuration Parameters

### Launch File Parameters

**vision.launch.py**
```xml
<!-- Camera mode -->
<arg name="camera_mode" default="subscribe" />
  <!-- Options: webcam, realsense, subscribe, file -->

<!-- Image source (for file mode) -->
<arg name="image_path" default="" />

<!-- Detection parameters -->
<arg name="sam_confidence_threshold" default="0.5" />
<arg name="clip_confidence_threshold" default="0.3" />

<!-- GPU settings -->
<arg name="use_gpu" default="true" />
<arg name="gpu_device" default="0" />
```

### Runtime Parameters

```bash
# Set parameters when launching
ros2 launch vision vision.launch.py \
  camera_mode:=webcam \
  sam_confidence_threshold:=0.7 \
  use_gpu:=true

# Or set at runtime
ros2 param set /sam_vision_pipeline_node sam_confidence_threshold 0.6
```

---

## Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| `NO_IMAGE` | Image path invalid or not provided | Check file path exists |
| `LOAD_ERROR` | Failed to load image | Verify image format (jpg, png) |
| `MODEL_ERROR` | Model initialization failed | Check GPU/CUDA availability |
| `NO_DETECTIONS` | No objects detected | Try lowering confidence threshold |
| `SERVICE_TIMEOUT` | Service call timed out | Check if pipeline node is running |

---

## Testing

### Unit Tests

```bash
cd ~/final_project_ws/src/vision
python3 -m pytest test/
```

### Integration Tests

```bash
# Full pipeline test
python3 vision_scripts/integration_test.py

# Service-level test
python3 vision_scripts/test_services.py

# Performance benchmark
python3 vision_scripts/benchmark_pipeline.py
```

---

## Advanced Usage

### Custom Object Categories

```python
# Define custom categories
categories = [
    "red_cup", "blue_cup", "green_bottle",
    "wooden_block", "metal_tool"
]

# Call classification with custom categories
request = ClassifyRegions.Request()
request.categories = categories
response = classify_client.call(request)
```

### Batch Processing

```python
import os

image_dir = "/path/to/images/"
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        results = detect_and_classify(image_path)
        # Process results...
```

### Visualization

```python
import cv2
from vision_scripts.view_detections import draw_detections

# Get detections
response = detect_client.call(request)

# Visualize
image = cv2.imread(image_path)
annotated = draw_detections(image, response.objects)
cv2.imshow('Detections', annotated)
cv2.waitKey(0)
```

---

## Support

For issues or questions:
1. Check the troubleshooting section in QUICK_START.md
2. Review architecture details in ARCHITECTURE.md
3. See example scripts in `vision_scripts/`
