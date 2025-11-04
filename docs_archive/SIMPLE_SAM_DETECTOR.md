# Simple SAM Detector - JSON Schema Export

## 📋 Overview

A clean, focused ROS2 vision node that:
- ✅ Subscribes to `/camera/image_raw` (sensor_msgs/Image)
- ✅ Uses `bridge.imgmsg_to_cv2(msg, 'bgr8')` for conversion
- ✅ Displays detections with OpenCV `cv2.imshow()`
- ✅ Exports detection data in structured JSON schema
- ✅ Two modes: single-shot or continuous detection

---

## 🚀 Quick Start

### 1. Run the Detector

**Continuous Mode** (auto-detects every frame):
```bash
cd /home/group11/final_project_ws
source install/setup.bash
ros2 run vision simple_sam_detector
```

**Single-Shot Mode** (only detects on service call):
```bash
ros2 run vision simple_sam_detector --single
```

### 2. Call Detection Service

```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

### 3. Test with Script

```bash
/home/group11/final_project_ws/src/vision/test_detection_json.sh
```

---

## 📊 JSON Schema Output

### Response Format

When you call the `/vision/detect_objects` service, you get:

```json
{
  "success": true,
  "detections": [
    {
      "image_id": "frame_000123",
      "detections": [
        {
          "class_name": "object",
          "confidence": 0.85,
          "bbox": [120, 200, 400, 800],
          "distance_cm": 125.6
        },
        {
          "class_name": "object",
          "confidence": 0.92,
          "bbox": [450, 300, 800, 650],
          "distance_cm": 243.2
        }
      ]
    }
  ],
  "summary": {
    "total_detections": 2,
    "average_distance_cm": 184.4,
    "timestamp": "2025-11-03T19:45:00Z"
  }
}
```

### Schema Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Service call success status |
| `detections[].image_id` | string | Frame identifier (e.g., "frame_000123") |
| `detections[].detections[]` | array | List of detected objects |
| `class_name` | string | Object class (default: "object") |
| `confidence` | float | Detection confidence (0.60-0.95) |
| `bbox` | [x1, y1, x2, y2] | Bounding box coordinates |
| `distance_cm` | float | Distance in centimeters (if depth available) |
| `summary.total_detections` | int | Total number of objects detected |
| `summary.average_distance_cm` | float | Average distance (if depth available) |
| `summary.timestamp` | string | ISO 8601 UTC timestamp |

---

## 🔧 Detection Pipeline

### Image Processing Flow

```
/camera/image_raw (ROS Image)
    ↓
bridge.imgmsg_to_cv2(msg, 'bgr8')
    ↓
OpenCV Processing:
  - Grayscale conversion
  - Gaussian blur (5x5)
  - Adaptive thresholding
  - Morphological operations (close + open)
  - Contour detection (RETR_EXTERNAL)
    ↓
Filtering:
  - Min area: 0.1% of image
  - Max area: 80% of image
  - Min box size: 20x20 pixels
    ↓
Detection Results:
  - Bounding boxes
  - Segmentation masks
  - Confidence scores
  - Distance estimation (if depth available)
    ↓
cv2.imshow() visualization
JSON schema export
```

---

## 👁️ Visualization

The OpenCV window shows:

- **Green boxes**: Bounding boxes around detected objects
- **Green overlay**: Semi-transparent segmentation masks
- **Labels**: `object: 0.85 (125.6cm)` (class, confidence, distance)
- **Top-left info**: Mode and detection count

**Window Controls:**
- ESC or 'q': Quit
- Window auto-updates at 30 Hz

---

## 📝 Example Service Calls

### Basic Call
```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

### Pretty JSON Output
```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger | \
  grep -A 1000 'message:' | sed '1d' | jq .
```

### Save to File
```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger | \
  grep -A 1000 'message:' | sed '1d' > detections.json
```

### Extract Just Bounding Boxes
```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger | \
  grep -A 1000 'message:' | sed '1d' | jq '.detections[0].detections[].bbox'
```

---

## 🎯 Use Cases

### 1. Single Detection Query
```bash
# Start in single-shot mode
ros2 run vision simple_sam_detector --single

# Call when needed
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

### 2. Continuous Monitoring
```bash
# Start in continuous mode (default)
ros2 run vision simple_sam_detector

# Optionally call service for JSON snapshot
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

### 3. Integration with Other Nodes
```python
import rclpy
from std_srvs.srv import Trigger
import json

# Create service client
client = node.create_client(Trigger, '/vision/detect_objects')
client.wait_for_service()

# Call service
request = Trigger.Request()
future = client.call_async(request)
rclpy.spin_until_future_complete(node, future)

# Parse JSON response
response = future.result()
if response.success:
    data = json.loads(response.message)
    detections = data['detections'][0]['detections']
    print(f"Found {len(detections)} objects")
```

---

## 🔍 Distance Estimation

If `/camera/depth/image_raw` is available, distance is calculated for each object:

- **Method**: Sample depth at object center point
- **Units**: Centimeters
- **Accuracy**: Depends on depth camera calibration
- **Fallback**: If depth unavailable, `distance_cm` field is omitted

---

## 🐛 Troubleshooting

### No OpenCV Window
```bash
# Check if node is running
ros2 node list | grep simple_sam_detector

# Check display
echo $DISPLAY
```

### No Detections
```bash
# Verify camera is publishing
ros2 topic hz /camera/image_raw

# Check image content
ros2 run rqt_image_view rqt_image_view /camera/image_raw

# Adjust detection parameters in simple_sam_detector.py:
# - min_area / max_area thresholds
# - Adaptive threshold block size
# - Morphology iterations
```

### Service Timeout
```bash
# Check if service exists
ros2 service list | grep detect_objects

# Ensure node is running
ros2 node info /simple_sam_detector
```

---

## 📦 Files

- **Main Node**: `/src/vision/vision/simple_sam_detector.py`
- **Test Script**: `/src/vision/test_detection_json.sh`
- **Setup Entry**: Added to `setup.py` as `simple_sam_detector`

---

## 🎓 Technical Details

### OpenCV Detection Algorithm

1. **Preprocessing**:
   - Convert BGR to grayscale
   - Apply Gaussian blur (reduces noise)

2. **Thresholding**:
   - Adaptive threshold (GAUSSIAN_C)
   - Block size: 11, constant: 2
   - Binary invert mode

3. **Morphological Filtering**:
   - Close operation (2 iterations) - connects nearby regions
   - Open operation (1 iteration) - removes small noise

4. **Contour Extraction**:
   - Find external contours only
   - Filter by area (0.1% - 80% of image)
   - Filter by bounding box size (min 20x20 pixels)

5. **Confidence Calculation**:
   - Based on circularity: `4π × area / perimeter²`
   - Range: 0.60 (irregular) to 0.95 (circular)

---

## 🔄 Comparison: Old vs New

| Feature | Old `sam_vision_pipeline` | New `simple_sam_detector` |
|---------|---------------------------|---------------------------|
| **Lines of code** | 726 | ~350 |
| **Focus** | Multiple services | Single detection |
| **Visualization** | Optional ROS topics | Always OpenCV window |
| **JSON Export** | ❌ | ✅ Structured schema |
| **Distance** | Complex pipeline | Direct depth sampling |
| **Setup** | Multiple dependencies | Self-contained |
| **Mode options** | Manual only | Continuous + single-shot |

---

## 💡 Tips

1. **View Live + Get JSON**: Run in continuous mode, OpenCV window shows live feed, call service for JSON snapshot
2. **Parse JSON in Python**: `import json; data = json.loads(response.message)`
3. **Chain with Other Nodes**: Use detection positions for robot navigation/grasping
4. **Adjust Thresholds**: Edit `_detect_objects()` method to tune for your scene
5. **Add Classification**: Replace `"object"` with actual class names from CLIP/other classifier

---

## 📄 License

Apache-2.0

---

## 🤝 Support

For issues or questions about the SAM detector:
1. Check logs: `ros2 node info /simple_sam_detector`
2. Verify topics: `ros2 topic list | grep camera`
3. Test service: `/src/vision/test_detection_json.sh`

**Happy Detecting! 🎯**
