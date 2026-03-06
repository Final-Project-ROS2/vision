# Unified Vision Pipeline - System Architecture

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER CALLS: /vision/run_unified_pipeline (std_srvs/srv/Trigger)    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: SAM DETECTION                                               │
│  Service: /vision/detect_objects (DetectObjects)                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Filter: confidence > 0.4 ✓ (line 633)                       │  │
│  │ • Returns: obj_id, bbox, confidence, iou_score, distance_cm    │  │
│  │ • Publishes: /vision/sam_detections (SAMDetections msg)        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  Output: List[{obj_id, bbox, confidence, iou_score, ...}]           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼ (wait 500ms for auto-classification)
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: CLIP CLASSIFICATION                                         │
│  Service: /vision/classify_bbox_filtered (Trigger)                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Auto-subscribes to /vision/sam_detections                    │  │
│  │ • Filter: confidence > 0.5 ✓                                   │  │
│  │ • Returns: region_id → {label, confidence}                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  Output: Dict[obj_id → {label, confidence, is_top1_accurate}]       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: GRASPNET DETECTION                                          │
│  Service: /vision/detect_grasp (Trigger)                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Uses SAM bboxes                                              │  │
│  │ • Returns: best grasp pose per object                          │  │
│  │ • Includes: pixel (u,v), quality, width, angle                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  Output: Dict[obj_id → {grasp_pixel, quality, width, angle}]        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼ (for each grasp_pixel)
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: PIXEL-TO-REAL CONVERSION                                    │
│  Service: /pixel_to_real (PixelToReal)                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Input: pixel (u, v)                                          │  │
│  │ • Output: world (x, y, z) in meters                            │  │
│  │ • Uses depth sensor + calibration                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  Output: Dict[obj_id → grasp_world{x, y, z}]                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: SCENE UNDERSTANDING                                         │
│  Service: /vision/understand_scene (Trigger)                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Uses SAM bboxes + CLIP labels                                │  │
│  │ • Analyzes spatial relations between objects                   │  │
│  │ • Returns: relations list per object                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  Output: Dict[obj_id → List[{target_id, target_label, relation}]]   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: INNER JOIN & UNIFIED JSON                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Join all data by obj_id:                                       │  │
│  │   {                                                            │  │
│  │     obj_id: 0,                                                 │  │
│  │     sam: {bbox, confidence, iou_score, ...},                   │  │
│  │     clip: {label, confidence, is_top1_accurate},               │  │
│  │     graspnet: {                                                │  │
│  │       grasp_pixel: {u, v},                                     │  │
│  │       grasp_world: {x, y, z},  ← FROM PIXEL-TO-REAL            │  │
│  │       quality, width, angle                                    │  │
│  │     },                                                          │  │
│  │     scene_understanding: {relations: [...]}                    │  │
│  │   }                                                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  Output: Complete unified JSON for all objects                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 7: SAVE & RETURN                                               │
│  • Save to: ~/unified_pipeline_outputs/unified_pipeline_*.json       │
│  • Return: JSON in service response                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Tables (INNER JOIN Example)

### Object 0 (green_cube)
```
┌──────────────────────────────────────────────────────────────────┐
│ obj_id: 0                                                         │
├──────────────────────────────────────────────────────────────────┤
│ SAM TABLE:                                                        │
│   bbox: [100, 150, 250, 300]                                      │
│   confidence: 0.85                                                │
│   ap_iou_score: 0.72                                              │
│   is_stable_detection: true                                       │
├──────────────────────────────────────────────────────────────────┤
│ CLIP TABLE:                                                       │
│   label: "green_cube"                                             │
│   confidence: 0.92                                                │
│   is_top1_accurate: true                                          │
├──────────────────────────────────────────────────────────────────┤
│ GRASPNET TABLE:                                                   │
│   grasp_pixel: {u: 175, v: 225}                                   │
│   grasp_world: {x: 0.45, y: -0.12, z: 0.82}  ← PIXEL-TO-REAL     │
│   quality_score: 0.88                                             │
│   grasp_width: 0.05                                               │
│   approach_angle: 45.0                                            │
├──────────────────────────────────────────────────────────────────┤
│ SCENE TABLE:                                                      │
│   relations: [                                                    │
│     {target_id: 1, target_label: "drill", relation: "left_of"},  │
│     {target_id: 2, target_label: "gear", relation: "near"}       │
│   ]                                                               │
└──────────────────────────────────────────────────────────────────┘
```

### Object 1 (drill)
```
┌──────────────────────────────────────────────────────────────────┐
│ obj_id: 1                                                         │
├──────────────────────────────────────────────────────────────────┤
│ SAM: bbox: [300, 100, 450, 280], confidence: 0.91, iou: 0.68     │
│ CLIP: label: "drill", confidence: 0.88, is_top1_accurate: true   │
│ GRASPNET: pixel: (375, 190), world: (0.62, -0.24, 0.83), ...     │
│ SCENE: relations: [{target_id: 0, relation: "right_of"}]         │
└──────────────────────────────────────────────────────────────────┘
```

## Service Dependencies

```
unified_pipeline.py (orchestrator)
    ├─→ /vision/detect_objects (DetectObjects)
    │   └─→ simple_sam_detector.py ✓ confidence > 0.4
    │
    ├─→ /vision/classify_bbox_filtered (Trigger)
    │   └─→ clip_classifier.py ✓ confidence > 0.5
    │
    ├─→ /vision/detect_grasp (Trigger)
    │   └─→ graspnet_detector.py
    │
    ├─→ /pixel_to_real (PixelToReal) [for each grasp]
    │   └─→ pixel_to_real.py (converts u,v → x,y,z)
    │
    └─→ /vision/understand_scene (Trigger)
        └─→ scene_understanding.py
```

## Node Startup Order

```
Time    Node                    Service
────────────────────────────────────────────────────────────────
0s      simple_sam_detector     /vision/detect_objects
2s      clip_classifier         /vision/classify_bbox_filtered
4s      graspnet_detector       /vision/detect_grasp
6s      scene_understanding     /vision/understand_scene
8s      pixel_to_real_service   /pixel_to_real
10s     unified_pipeline        /vision/run_unified_pipeline
────────────────────────────────────────────────────────────────
        ↓
15s+    READY TO RUN PIPELINE
```

## Confidence Filters

```
┌────────────┬────────────────┬──────────────────────────────┐
│ Component  │ Threshold      │ Location                      │
├────────────┼────────────────┼──────────────────────────────┤
│ SAM        │ conf > 0.4 ✓   │ simple_sam_detector.py:633    │
│ CLIP       │ conf > 0.5 ✓   │ clip_classifier.py (filtered) │
│ GraspNet   │ quality > 0.0  │ No filter (best pose)         │
│ Scene      │ N/A            │ Always computed               │
└────────────┴────────────────┴──────────────────────────────┘
```

## Output JSON Schema

```json
{
  "pipeline": "unified_vision_pipeline",
  "timestamp": "2025-11-17T15:30:45.123Z",
  "total_objects": N,
  "objects": [
    {
      "obj_id": 0,  ← PRIMARY KEY (INNER JOIN)
      "sam": {
        "object_id": "obj_0",
        "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
        "confidence": float (> 0.4),
        "ap_iou_score": float (0.0-1.0),
        "is_stable_detection": bool (IoU >= 0.5)
      },
      "clip": {
        "label": string,
        "confidence": float (> 0.5),
        "is_top1_accurate": bool (conf >= 0.5)
      },
      "graspnet": {
        "grasp_pixel": {"u": int, "v": int},
        "grasp_world": {"x": float, "y": float, "z": float},
        "quality_score": float (0.0-1.0),
        "grasp_width": float (meters),
        "approach_angle": float (degrees)
      },
      "scene_understanding": {
        "relations": [
          {
            "target_id": int,
            "target_label": string,
            "relation": string (left_of, right_of, near, ...)
          }
        ]
      }
    }
  ],
  "summary": {
    "sam_detections": int,
    "clip_classifications": int,
    "graspnet_detections": int,
    "scene_relations": int
  }
}
```

## Usage Commands

### Start System
```bash
# Method 1: Launch all at once
ros2 launch vision unified_pipeline.launch.py

# Method 2: Manual (separate terminals)
ros2 run vision simple_sam_detector
ros2 run vision clip_classifier
ros2 run vision graspnet_detector
ros2 run vision scene_understanding
ros2 run vision pixel_to_real_service
ros2 run vision unified_pipeline
```

### Run Pipeline
```bash
# Wait 15 seconds after launch, then:
ros2 service call /vision/run_unified_pipeline std_srvs/srv/Trigger
```

### View Results
```bash
# Latest JSON file
cat ~/unified_pipeline_outputs/$(ls -t ~/unified_pipeline_outputs/ | head -1) | jq .

# Pretty print with jq
cat ~/unified_pipeline_outputs/unified_pipeline_*.json | jq '.objects[] | {obj_id, sam: .sam.bbox, clip: .clip.label, grasp: .graspnet.grasp_world}'
```

### Test Individual Services
```bash
# Test SAM (with confidence > 0.4)
ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects

# Test CLIP (with confidence > 0.5)
ros2 service call /vision/classify_bbox_filtered std_srvs/srv/Trigger

# Test GraspNet
ros2 service call /vision/detect_grasp std_srvs/srv/Trigger

# Test Scene Understanding
ros2 service call /vision/understand_scene std_srvs/srv/Trigger

# Test Pixel-to-Real
ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"
```

## Performance Metrics

```
Total Pipeline Time: ~1.1-1.9 seconds per frame
├─ SAM Detection:        200-500ms (with filter)
├─ CLIP Classification:  300-400ms (with filter)
├─ GraspNet Detection:   500-800ms
├─ Pixel-to-Real:        10-20ms × N objects
├─ Scene Understanding:  100-200ms
└─ JSON Building:        <10ms
```

## Success Criteria ✅

- [✓] SAM filters confidence > 0.4
- [✓] CLIP filters confidence > 0.5
- [✓] All data joined by obj_id (INNER JOIN)
- [✓] Grasp coordinates in pixel AND world space
- [✓] Spatial relations computed
- [✓] JSON output saved to file
- [✓] Service returns unified response
- [✓] Build successful
- [✓] Documentation complete



# Vision System Architecture

## System Overview

```
┌─────────────────┐
│  Camera Input   │
│  (3 modes)      │
└────────┬────────┘
         │
         ↓
┌──────────────────────────────────────────┐
│         Vision Pipeline Node             │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Stage 1: SAM Detector             │ │
│  │  - Segment Anything Model          │ │
│  │  - Object segmentation & masks     │ │
│  └────────────┬───────────────────────┘ │
│               ↓                          │
│  ┌────────────────────────────────────┐ │
│  │  Stage 2: CLIP Classifier          │ │
│  │  - Semantic classification         │ │
│  │  - Label assignment                │ │
│  └────────────┬───────────────────────┘ │
│               ↓                          │
│  ┌────────────────────────────────────┐ │
│  │  Stage 3: GraspNet Predictor       │ │
│  │  - 6D grasp pose generation        │ │
│  │  - Grasp quality scores            │ │
│  └────────────┬───────────────────────┘ │
│               ↓                          │
│  ┌────────────────────────────────────┐ │
│  │  Stage 4: Scene Graph Builder      │ │
│  │  - Spatial relationships           │ │
│  │  - Object hierarchy                │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

---

## Components

### 1. Camera Service Node
**File:** `vision/camera_service_node.py`

**Modes:**
- `webcam` - Capture from /dev/video device
- `realsense` - Intel RealSense camera with depth
- `subscribe` - Subscribe to existing /camera/image_raw topic
- `file` - Load from static image file

**Published Topics:**
- `/camera/image_raw` (sensor_msgs/Image)
- `/camera/depth` (sensor_msgs/Image) - when using RealSense

### 2. SAM Vision Pipeline Node
**File:** `vision/sam_vision_pipeline_node.py`

**Core Pipeline:**
- SAM (Segment Anything Model) for detection
- CLIP (Contrastive Language-Image Pre-training) for classification
- GraspNet for grasp pose estimation
- Scene graph builder for spatial relationships

**Services Provided:**
- `/detect_objects` (vision/srv/DetectObjects)
- `/classify_objects` (vision/srv/ClassifyRegions)
- `/generate_grasps` (vision/srv/GenerateGrasps)
- `/build_scene_graph` (vision/srv/BuildSceneGraph)

### 3. Pipeline Modules

#### SAM Detector
**File:** `vision/simple_sam_detector.py`
- Uses facebook/sam-vit-huge model
- Automatic mask generation
- Configurable confidence thresholds

#### CLIP Classifier
**File:** `vision/clip_classifier.py`
- Uses openai/clip-vit-base-patch32
- Custom category support
- Batch processing optimization

#### SAM+CLIP Integration
**File:** `vision/sam_clip_pipeline.py`
- Coordinates SAM detection + CLIP classification
- Handles image preprocessing
- JSON output formatting

---

## Data Flow

### Service Call Sequence

```
Client Request
     ↓
1. detect_objects (image_path)
     ↓
   [SAM processes image]
     ↓
   Returns: masks, bboxes, scores
     ↓
2. classify_objects (image, regions)
     ↓
   [CLIP classifies each region]
     ↓
   Returns: labels, confidences
     ↓
3. generate_grasps (image, masks, labels)
     ↓
   [GraspNet predicts grasps]
     ↓
   Returns: 6D poses, scores
     ↓
4. build_scene_graph (objects, positions)
     ↓
   [Scene graph builder]
     ↓
   Returns: relationships, hierarchy
```

---

## Configuration

### Launch Parameters

**vision.launch.py**
```python
camera_mode='subscribe'  # webcam, realsense, subscribe, file
image_path='/path/to/image.jpg'  # for file mode
capture_single=True  # for webcam: capture once and exit
```

### Model Configuration

Models are downloaded automatically to `~/.cache/torch/hub/`

- **SAM:** facebook/sam-vit-huge (~2.4GB)
- **CLIP:** openai/clip-vit-base-patch32 (~600MB)
- **GraspNet:** (custom weights if available)

---

## Message Types

### Custom Messages

**DetectionResult.msg**
```
sensor_msgs/Image image
string[] labels
float32[] confidences
int32[] bbox_x
int32[] bbox_y
int32[] bbox_w
int32[] bbox_h
uint8[] masks
```

**SemanticObject.msg**
```
string label
float32 confidence
geometry_msgs/Pose pose
geometry_msgs/Point[] mask_points
```

**GraspPose.msg**
```
geometry_msgs/Pose pose
float32 score
string object_label
```

**SceneGraph.msg**
```
SemanticObject[] objects
SpatialRelation[] relations
```

---

## File Structure

```
vision/
├── vision/                          # Main package code
│   ├── sam_vision_pipeline_node.py # Main pipeline node
│   ├── camera_service_node.py      # Camera interface
│   ├── simple_sam_detector.py      # SAM wrapper
│   ├── clip_classifier.py          # CLIP wrapper
│   └── sam_clip_pipeline.py        # Integrated pipeline
├── vision_scripts/                  # Test & utility scripts
│   ├── test_services.py            # Service testing
│   ├── integration_test.py         # Full pipeline test
│   └── view_detections.py          # Visualization
├── launch/                          # ROS2 launch files
│   ├── vision.launch.py            # Main launcher
│   └── vision_with_camera.launch.py
├── msg/                             # Custom message definitions
├── srv/                             # Custom service definitions
└── docs/                            # Documentation (this folder)
```

---

## Performance Considerations

### GPU Acceleration
- SAM and CLIP use CUDA if available
- Falls back to CPU automatically
- Check with: `python3 -c "import torch; print(torch.cuda.is_available())"`

### Memory Usage
- SAM: ~4GB GPU memory
- CLIP: ~2GB GPU memory
- Total: ~6-8GB recommended

### Processing Time (GPU)
- SAM detection: 2-5 seconds
- CLIP classification: 0.1-0.5 seconds per object
- Full pipeline: 3-10 seconds depending on object count

---

## Integration with Robot System

### Connecting to Gazebo

```bash
# Terminal 1: Start Gazebo
ros2 launch ur_yt_sim spawn_ur5_camera_gripper_moveit.launch.py

# Terminal 2: Start vision pipeline
ros2 launch vision vision.launch.py
```

### Using Detection Results

```python
# In your robot control node
from vision.srv import DetectObjects

# Call detection service
client = self.create_client(DetectObjects, '/detect_objects')
response = client.call(request)

# Use results
for obj in response.objects:
    label = obj.label
    pose = obj.pose
    # Plan grasp, navigate, etc.
```

---

## Development

### Adding New Object Categories

Edit `vision/clip_classifier.py`:
```python
CATEGORIES = [
    "bottle", "cup", "bowl",
    "your_new_category"  # Add here
]
```

### Extending the Pipeline

1. Create new module in `vision/`
2. Add service definition in `srv/`
3. Register service in `sam_vision_pipeline_node.py`
4. Update launch file if needed

---

## References

- SAM: https://github.com/facebookresearch/segment-anything
- CLIP: https://github.com/openai/CLIP
- ROS2: https://docs.ros.org/en/humble/
