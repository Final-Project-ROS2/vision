# SAM + CLIP Pipeline

A dedicated ROS2 node that combines SAM object detection with CLIP classification.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    sam_clip_pipeline.py                     │
├─────────────────────────────────────────────────────────────┤
│  1. Subscribe to /camera/image_raw (capture ONE frame)     │
│  2. Call /vision/detect_objects (SAM service)              │
│  3. Parse bounding boxes from SAM JSON response            │
│  4. Classify each region with CLIP                         │
│  5. Return combined results                                 │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Terminal 1: Start SAM Detector
```bash
ros2 run vision simple_sam_detector
```

### Terminal 2: Start SAM+CLIP Pipeline
```bash
ros2 run vision sam_clip_pipeline
```

### Terminal 3: Call the Pipeline Service
```bash
ros2 service call /vision/classify_detect std_srvs/srv/Trigger
```

## Features

### 1. **Single Frame Capture**
- Subscribes to `/camera/image_raw`
- Captures **ONE frame** at startup
- Uses this frame for all subsequent processing

### 2. **SAM Integration**
- Calls `/vision/detect_objects` service
- Prints full SAM JSON response for debugging
- Extracts bounding boxes from response

### 3. **Detailed Logging**
- Shows captured frame details
- Prints complete SAM JSON response
- Logs each bounding box extraction
- Shows CLIP classification for each region

### 4. **CLIP Classification**
- Classifies each detected region
- Returns top predictions with confidence scores
- Supports custom labels

## Output Format

```json
{
  "pipeline": "sam_clip",
  "success": true,
  "model": "openai/clip-vit-base-patch32",
  "input": {
    "image_id": "frame_000001",
    "image_shape": [480, 640, 3],
    "num_regions": 3,
    "candidate_labels": ["cobot", "green_cube", "drill", ...]
  },
  "output": {
    "classified_regions": [
      {
        "region_id": 0,
        "bbox": [100, 150, 250, 300],
        "top_prediction": {
          "label": "drill",
          "confidence": 0.89
        },
        "all_predictions": [...]
      }
    ],
    "summary": {
      "total_regions": 3,
      "processing_time_ms": 1250
    }
  },
  "metadata": {
    "timestamp": "2025-11-04T12:34:56.789Z",
    "device": "cuda"
  }
}
```

## Advantages Over Original clip_classifier.py

1. **Cleaner Architecture**: Dedicated pipeline node
2. **Single Responsibility**: Only handles SAM→CLIP pipeline
3. **Better Debugging**: Comprehensive logging at each step
4. **Frame Guarantee**: Captures ONE frame, ensures consistency
5. **Service-Focused**: Designed specifically for service calls

## Dependencies

- `ros2` (ROS2 Humble or later)
- `simple_sam_detector` (must be running)
- `torch`, `transformers`, `pillow` (CLIP)
- Camera publishing to `/camera/image_raw`

## Troubleshooting

### "No frame captured yet"
- Wait 1-2 seconds after starting the node
- Check if camera is publishing: `ros2 topic echo /camera/image_raw`

### "SAM detector service not available"
- Make sure SAM detector is running: `ros2 run vision simple_sam_detector`
- Check service: `ros2 service list | grep detect_objects`

### "No bounding boxes extracted"
- Check SAM JSON response in terminal output
- Verify objects are visible in camera view
- Check SAM detector is finding objects

## Comparison: sam_clip_pipeline vs clip_classifier

| Feature | sam_clip_pipeline.py | clip_classifier.py |
|---------|---------------------|-------------------|
| Purpose | SAM→CLIP pipeline | CLIP classification + optional SAM |
| Frame Handling | Capture ONE frame | Continuous updates |
| Services | `/vision/classify_detect` | `/vision/classify_all`, `/vision/classify_detect` |
| Logging | Detailed step-by-step | Standard logging |
| Architecture | Clean, focused | All-in-one |
| Best For | Service calls | Live classification |

## Next Steps

To build and run:
```bash
cd /home/group11/final_project_ws
colcon build --packages-select vision
source install/setup.bash
ros2 run vision sam_clip_pipeline
```
