# Integrated Vision Pipeline: SAM + CLIP

## Overview

The integrated pipeline combines **SAM object detection** with **CLIP classification** to detect and classify objects in camera images.

```
/camera/image_raw → SAM Detector → CLIP Classifier → Classified Objects
```

## Architecture

### Components

1. **Camera Source**: `/camera/image_raw` topic
2. **SAM Detector Node**: `simple_sam_detector` - detects objects and generates bounding boxes
3. **CLIP Classifier Node**: `clip_classifier` - classifies detected regions with semantic labels

### Data Flow

```
┌─────────────────┐
│ Camera          │
│ /camera/        │
│ image_raw       │
└────────┬────────┘
         │ Image
         ▼
┌─────────────────────────────────┐
│ Service: /vision/classify_detect│
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ SAM Detector    │ ──┐
│ Service Call    │   │ Bounding Boxes
└─────────────────┘   │ [x1,y1,x2,y2]...
         ▲            │
         │            ▼
         │    ┌─────────────────┐
         │    │ CLIP Classifier │
         │    │ Per-Region      │
         │    └────────┬────────┘
         │             │
         │             ▼
    ┌────┴────────────────────┐
    │ JSON Response with      │
    │ Detected + Classified   │
    │ Objects                 │
    └─────────────────────────┘
```

## Usage

### 1. Start Both Nodes

**Terminal 1: Start SAM Detector**
```bash
source install/setup.bash
ros2 run vision simple_sam_detector
```

**Terminal 2: Start CLIP Classifier**
```bash
source install/setup.bash
ros2 run vision clip_classifier
```

### 2. Run Integrated Pipeline

**Call the service:**
```bash
ros2 service call /vision/classify_detect std_srvs/srv/Trigger
```

**Or use the test script:**
```bash
cd /home/group11/final_project_ws/src/vision
./test_integrated_pipeline.sh
```

## Services

### `/vision/classify_detect`

**Type:** `std_srvs/srv/Trigger`

**Request:** Empty (takes current camera image)

**Response:**
```json
{
  "pipeline": "clip_with_detection",
  "model": "openai/clip-vit-base-patch32",
  "input": {
    "image_path": "frame_000123",
    "num_regions": 3,
    "candidate_labels": [
      "cobot",
      "green_cube",
      "drill",
      "gear",
      "monkey_wrench",
      "piston_rod",
      "washer"
    ]
  },
  "output": {
    "classified_regions": [
      {
        "region_id": 0,
        "bbox": [100, 150, 300, 400],
        "top_prediction": {
          "label": "drill",
          "confidence": 0.87
        },
        "all_predictions": [
          {"label": "drill", "confidence": 0.87},
          {"label": "piston_rod", "confidence": 0.08},
          ...
        ]
      },
      ...
    ],
    "summary": {
      "total_regions": 3,
      "processing_time_ms": 234
    },
    "metadata": {
      "timestamp": "2025-11-04T12:34:56.789Z",
      "device": "cuda"
    }
  },
  "sam_detection": {
    "total_detections": 3,
    "bboxes": [[100, 150, 300, 400], ...]
  }
}
```

## Classification Labels

Default labels (from candidate_labels):
- `cobot` - Collaborative robot
- `green_cube` - Green cube object
- `drill` - Power drill tool
- `gear` - Mechanical gear
- `monkey_wrench` - Adjustable wrench
- `piston_rod` - Piston rod component
- `washer` - Washer component

**Custom labels:**
```bash
ros2 run vision clip_classifier --labels "cat,dog,car,airplane"
```

## Visualization

### SAM Detector Window
- **Green bounding boxes** around detected objects
- Object ID and confidence score

### CLIP Classifier Window
- **Yellow bounding boxes** around classified regions
- Region ID and predicted label
- Confidence percentage

## Pipeline Workflow

1. **Image Capture**: Latest frame from `/camera/image_raw` is stored
2. **Service Call**: User calls `/vision/classify_detect`
3. **SAM Detection**: CLIP node calls `/vision/detect_objects` service
4. **Bbox Extraction**: Bounding boxes extracted from SAM response
5. **Region Cropping**: Each bbox region is cropped from the image
6. **CLIP Classification**: Each region is classified independently
7. **Result Aggregation**: All classifications combined into JSON response
8. **Visualization**: Yellow boxes with labels displayed in OpenCV window

## Error Handling

### "SAM detector service not available"
```bash
# Check if simple_sam_detector is running
ros2 service list | grep detect_objects

# If not, start it:
ros2 run vision simple_sam_detector
```

### "No objects detected"
- Adjust lighting or camera position
- Check SAM detector parameters in `simple_sam_detector.py`
- Verify objects are in camera view

### "CLIP model not available"
```bash
# Install dependencies
pip install torch transformers pillow
```

## Performance

- **SAM Detection**: ~100-500ms (depends on image size and number of objects)
- **CLIP Classification**: ~50-150ms per region (GPU) or ~500-1000ms (CPU)
- **Total Pipeline**: Typically 200-1000ms for 1-5 objects

## Examples

### Example 1: Detect and Classify All Objects
```bash
ros2 service call /vision/classify_detect std_srvs/srv/Trigger
```

### Example 2: Continuous Pipeline (Python)
```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

node = rclpy.create_node('pipeline_test')
client = node.create_client(Trigger, '/vision/classify_detect')
client.wait_for_service()

request = Trigger.Request()
future = client.call_async(request)
rclpy.spin_until_future_complete(node, future)

result = future.result()
print(result.message)
```

## Troubleshooting

### Pipeline is slow
- Use GPU: CUDA will be automatically detected
- Reduce image resolution
- Reduce number of candidate labels

### Incorrect classifications
- Add more specific labels to candidate_labels
- Adjust labels to match your objects
- Use more descriptive label names

### Service timeout
- Increase timeout in classify_detect_callback (default 10s)
- Check if SAM detector is responsive

## Integration with Other Nodes

### Grasp Planning Integration
```python
# Call pipeline
response = classify_detect_client.call(Trigger.Request())
data = json.loads(response.message)

# Extract classified objects
for region in data['output']['classified_regions']:
    if region['top_prediction']['label'] == 'drill':
        bbox = region['bbox']
        # Use bbox for grasp planning
        plan_grasp(bbox, 'drill')
```

### Scene Understanding
```python
# Get all objects in scene
response = classify_detect_client.call(Trigger.Request())
data = json.loads(response.message)

scene_objects = [
    region['top_prediction']['label']
    for region in data['output']['classified_regions']
]

print(f"Scene contains: {', '.join(scene_objects)}")
```

## Files

- `vision/clip_classifier.py` - Main CLIP classifier node
- `vision/simple_sam_detector.py` - SAM detection node
- `test_integrated_pipeline.sh` - Test script
- `INTEGRATED_PIPELINE.md` - This documentation

## See Also

- [CLIP_CLASSIFIER.md](CLIP_CLASSIFIER.md) - CLIP node documentation
- [SIMPLE_SAM_DETECTOR.md](SIMPLE_SAM_DETECTOR.md) - SAM detector documentation
- [QUICK_USAGE.md](QUICK_USAGE.md) - Quick start guide
