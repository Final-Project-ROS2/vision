# Unified Vision Pipeline

A comprehensive ROS2 vision pipeline that integrates SAM detection, CLIP classification, GraspNet grasp detection, and Scene Understanding into a single unified output with INNER JOIN on `obj_id`.

## Overview

The Unified Pipeline orchestrates all vision components and produces a structured JSON output with the following data for each detected object:

### Output Structure (INNER JOIN by obj_id)

```json
{
  "pipeline": "unified_vision_pipeline",
  "timestamp": "2025-11-17T12:34:56.789Z",
  "total_objects": 5,
  "objects": [
    {
      "obj_id": 0,
      "sam": {
        "object_id": "obj_0",
        "bbox": {
          "x1": 100,
          "y1": 150,
          "x2": 250,
          "y2": 300
        },
        "confidence": 0.85,
        "ap_iou_score": 0.72,
        "is_stable_detection": true
      },
      "clip": {
        "label": "green_cube",
        "confidence": 0.92,
        "is_top1_accurate": true
      },
      "graspnet": {
        "grasp_pixel": {
          "u": 175,
          "v": 225
        },
        "grasp_world": {
          "x": 0.45,
          "y": -0.12,
          "z": 0.82
        },
        "quality_score": 0.88,
        "grasp_width": 0.05,
        "approach_angle": 45.0
      },
      "scene_understanding": {
        "relations": [
          {
            "target_id": 1,
            "target_label": "drill",
            "relation": "left_of"
          },
          {
            "target_id": 2,
            "target_label": "gear",
            "relation": "near"
          }
        ]
      }
    }
  ],
  "summary": {
    "sam_detections": 5,
    "clip_classifications": 5,
    "graspnet_detections": 5,
    "scene_relations": 12
  }
}
```

## Data Schema

### SAM (Segment Anything Model)
| Field | Type | Description |
|-------|------|-------------|
| `obj_id` | int | Object identifier (0-indexed) |
| `object_id` | string | Object name (e.g., "obj_0") |
| `bbox` | object | Bounding box {x1, y1, x2, y2} |
| `confidence` | float | Detection confidence (0.0-1.0) |
| `ap_iou_score` | float | Average Precision IoU score (≥0.5 for stable) |
| `is_stable_detection` | bool | True if IoU ≥ 0.5 (COCO AP threshold) |

### CLIP (Classification)
| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Object class label |
| `confidence` | float | Classification confidence (0.0-1.0) |
| `is_top1_accurate` | bool | True if confidence ≥ 0.5 |

### GraspNet (Grasp Detection)
| Field | Type | Description |
|-------|------|-------------|
| `grasp_pixel` | object | Pixel coordinates {u, v} |
| `grasp_world` | object | World coordinates {x, y, z} in meters |
| `quality_score` | float | Grasp quality (0.0-1.0) |
| `grasp_width` | float | Gripper width in meters |
| `approach_angle` | float | Approach angle in degrees |

### Scene Understanding (Spatial Relations)
| Field | Type | Description |
|-------|------|-------------|
| `relations` | array | List of spatial relations with other objects |
| `target_id` | int | Target object ID |
| `target_label` | string | Target object label |
| `relation` | string | Spatial relation (e.g., "left_of", "near", "above") |

## Setup

### Prerequisites
All vision nodes must be running:

```bash
# Terminal 1: SAM Detector
ros2 run vision simple_sam_detector

# Terminal 2: CLIP Classifier
ros2 run vision clip_classifier

# Terminal 3: GraspNet Detector
ros2 run vision graspnet_detector

# Terminal 4: Scene Understanding
ros2 run vision scene_understanding

# Terminal 5: Pixel-to-Real Service
ros2 run vision pixel_to_real_service

# Terminal 6: Unified Pipeline
ros2 run vision unified_pipeline
```

### Build Package
```bash
cd ~/final_project_ws
colcon build --packages-select vision --symlink-install
source install/setup.bash
```

## Usage

### Run Unified Pipeline
```bash
ros2 service call /vision/run_unified_pipeline std_srvs/srv/Trigger
```

### Example Response
```json
{
  "success": true,
  "message": "{...unified JSON...}"
}
```

### Output Files
Results are saved to: `~/unified_pipeline_outputs/unified_pipeline_YYYYMMDD_HHMMSS.json`

## Pipeline Flow

```
1. SAM Detection
   ↓ (publishes /vision/sam_detections)
   
2. CLIP Classification (auto-subscribes to SAM)
   ↓ (filters confidence > 0.5)
   
3. GraspNet Detection (uses SAM bboxes)
   ↓ (finds grasp poses)
   
4. Pixel-to-Real Conversion (converts grasp pixels to world coords)
   ↓
   
5. Scene Understanding (analyzes spatial relations)
   ↓
   
6. Unified JSON Output (INNER JOIN by obj_id)
```

## Service Endpoints

| Service | Type | Description |
|---------|------|-------------|
| `/vision/run_unified_pipeline` | Trigger | Run complete pipeline |
| `/vision/detect_objects` | DetectObjects | SAM detection (with confidence > 0.4) |
| `/vision/classify_bbox_filtered` | Trigger | CLIP classification (confidence > 0.5) |
| `/vision/detect_grasp` | Trigger | GraspNet detection |
| `/vision/understand_scene` | Trigger | Scene analysis |
| `/pixel_to_real` | PixelToReal | Pixel-to-world conversion |

## Features

### INNER JOIN
- Only objects with complete data (SAM + CLIP + GraspNet + Scene) are included
- Missing data is filled with default values (e.g., "unknown" label)
- Ensures data consistency across all pipeline stages

### Confidence Filtering
- **SAM**: Only detections with confidence > 0.4 (set in `simple_sam_detector.py`)
- **CLIP**: Only classifications with confidence > 0.5 (filtered in `classify_bbox_filtered`)
- **GraspNet**: Best grasp pose per object (sorted by quality)

### Coordinate Systems
- **Pixel**: Image coordinates (u, v) - origin at top-left
- **World**: Real-world coordinates (x, y, z) in meters - calibrated via `/pixel_to_real`

### Spatial Relations
Supported relation types:
- `left_of`, `right_of`
- `above`, `below`
- `in_front_of`, `behind`
- `near`, `far_from`
- `touching`
- `aligned_horizontal`, `aligned_vertical`

## Integration with Existing Nodes

### SAM Detector (`simple_sam_detector.py`)
- ✅ Already filters confidence > 0.4
- ✅ Returns structured data via `/vision/detect_objects` service
- ✅ Includes IoU scores for AP@0.5 metric

### CLIP Classifier (`clip_classifier.py`)
- ✅ Auto-subscribes to `/vision/sam_detections`
- ✅ Filters results via `/vision/classify_bbox_filtered` (confidence > 0.5)
- ✅ Returns JSON with region_id → label mapping

### GraspNet Detector (`graspnet_detector.py`)
- ✅ Detects grasps for all SAM bounding boxes
- ✅ Returns best grasp pose per object
- ✅ Uses pixel coordinates from SAM bboxes

### Scene Understanding (`scene_understanding.py`)
- ✅ Analyzes spatial relations between objects
- ✅ Uses SAM bboxes and CLIP labels
- ✅ Returns relation graph

### Pixel-to-Real Service (`pixel_to_real.py`)
- ✅ Converts pixel (u, v) to world (x, y, z)
- ✅ Calibrated for camera setup
- ✅ Uses depth sensor for z-coordinate

## Example Output

See `~/unified_pipeline_outputs/` for saved JSON files.

Example with 3 objects:
```json
{
  "pipeline": "unified_vision_pipeline",
  "timestamp": "2025-11-17T15:30:45.123Z",
  "total_objects": 3,
  "objects": [
    {
      "obj_id": 0,
      "sam": {...},
      "clip": {"label": "green_cube", "confidence": 0.95, "is_top1_accurate": true},
      "graspnet": {"grasp_world": {"x": 0.5, "y": 0.0, "z": 0.8}, ...},
      "scene_understanding": {"relations": [{"target_id": 1, "relation": "left_of"}]}
    },
    {
      "obj_id": 1,
      "sam": {...},
      "clip": {"label": "drill", "confidence": 0.88, "is_top1_accurate": true},
      "graspnet": {...},
      "scene_understanding": {"relations": [{"target_id": 0, "relation": "right_of"}]}
    },
    {
      "obj_id": 2,
      "sam": {...},
      "clip": {"label": "gear", "confidence": 0.92, "is_top1_accurate": true},
      "graspnet": {...},
      "scene_understanding": {"relations": [{"target_id": 0, "relation": "near"}]}
    }
  ],
  "summary": {
    "sam_detections": 3,
    "clip_classifications": 3,
    "graspnet_detections": 3,
    "scene_relations": 6
  }
}
```

## Troubleshooting

### Service Not Available
Ensure all required nodes are running:
```bash
ros2 node list | grep vision
```

Expected output:
```
/clip_classifier
/graspnet_detector
/scene_understanding
/simple_sam_detector
/unified_pipeline
```

### Missing Data
Check individual services:
```bash
ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects
ros2 service call /vision/classify_bbox_filtered std_srvs/srv/Trigger
ros2 service call /vision/detect_grasp std_srvs/srv/Trigger
ros2 service call /vision/understand_scene std_srvs/srv/Trigger
```

### No Objects Detected
- Check camera feed: `rqt_image_view /camera/image_raw`
- Verify confidence threshold in `simple_sam_detector.py` (line 633)
- Ensure objects are visible and have sufficient contrast

## Performance

Typical processing time per frame:
- SAM Detection: ~200-500ms
- CLIP Classification: ~300-400ms
- GraspNet Detection: ~500-800ms
- Scene Understanding: ~100-200ms
- **Total**: ~1.1-1.9 seconds per frame

## Future Enhancements

- [ ] Parallel service calls for faster processing
- [ ] Object tracking across multiple frames
- [ ] Grasp success prediction
- [ ] 3D scene reconstruction
- [ ] Multi-camera support

## License

Apache-2.0
