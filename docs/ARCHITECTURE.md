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
