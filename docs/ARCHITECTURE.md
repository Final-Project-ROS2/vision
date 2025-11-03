# Vision System Architecture

## System Overview

The vision system consists of two separate ROS2 nodes that work together:

```
┌─────────────────────────────────────────────────────────────┐
│                     VISION SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────┐      ┌──────────────────────┐     │
│  │  Camera Service      │      │  SAM Vision Pipeline │     │
│  │  Node                │─────▶│  Node               │     │
│  └──────────────────────┘      └──────────────────────┘     │
│           │                              │                  │
│           │ Opens Camera                 │ Processes Images │
│           │ via CV Bridge                │ with AI Models   │
│           │                              │                  │
│           ▼                              ▼                  │
│  ┌──────────────────────┐      ┌──────────────────────┐     │
│  │  /camera/image_raw   │      │  Vision Services     │     │
│  │  /camera/depth       │      │  - detect_objects    │     │
│  │  /camera/info        │      │  - classify_objects  │     │
│  └──────────────────────┘      │  - get_positions     │     │
│                                │  - generate_grasps   │     │
│                                │  - build_scene_graph │     │
│                                └──────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Node 1: Camera Service

**File:** `vision/camera_service_node.py`

**Purpose:** Opens webcam/depth camera hardware and publishes images via CV Bridge

**Features:**
- Supports RGB webcams (USB, built-in)
- Supports Intel RealSense depth cameras
- Supports static image files for testing
- Configurable resolution and FPS
- Auto-start capability

**Topics Published:**
- `/camera/image_raw` (sensor_msgs/Image) - RGB stream
- `/camera/depth/image_raw` (sensor_msgs/Image) - Depth stream
- `/camera/camera_info` (sensor_msgs/CameraInfo) - Camera parameters

**Services Provided:**
- `/camera/start` (std_srvs/Trigger) - Start streaming
- `/camera/stop` (std_srvs/Trigger) - Stop streaming
- `/camera/reset` (std_srvs/Trigger) - Reset connection

**Parameters:**
- `camera_type`: 'webcam', 'realsense', 'file'
- `camera_id`: Device ID (default: 0)
- `image_file`: Path to image/video file
- `width`: Resolution width (default: 640)
- `height`: Resolution height (default: 480)
- `fps`: Frame rate (default: 30.0)
- `auto_start`: Auto-start on launch (default: true)

**Usage:**
```bash
# Webcam
ros2 run vision camera_service

# RealSense
ros2 run vision camera_service --ros-args -p camera_type:=realsense

# Test image
ros2 run vision camera_service --ros-args \
  -p camera_type:=file \
  -p image_file:=Final-proj/src/arrange.jpg
```

---

## Node 2: SAM Vision Pipeline

**File:** `vision/sam_vision_pipeline_node.py`

**Purpose:** Processes camera images for object detection, classification, and scene understanding

**Pipeline Stages:**
1. **SAM Detection** - Segment Anything Model for object segmentation
2. **CLIP Classification** - OpenAI CLIP for semantic tagging
3. **GraspNet** - 6D grasp pose generation
4. **Scene Graph** - Spatial relationship understanding

**Topics Subscribed:**
- `/camera/image_raw` (sensor_msgs/Image)
- `/camera/depth/image_raw` (sensor_msgs/Image)
- `/camera/camera_info` (sensor_msgs/CameraInfo)

**Topics Published:**
- `/vision/debug_image` (sensor_msgs/Image) - Visualization
- `/vision/grasp_poses` (geometry_msgs/PoseStamped) - Grasp poses

**Services Provided:**
1. `/vision/detect_objects` - Run SAM object detection
2. `/vision/classify_objects` - Run CLIP classification
3. `/vision/generate_grasps` - Generate 6D grasp poses
4. `/vision/get_positions` - Extract 3D object positions
5. `/vision/build_scene_graph` - Build scene relationships
6. `/vision/process_scene` - Run full pipeline
7. `/vision/reset_pipeline` - Reset cached state

**Parameters:**
- `auto_process`: Auto-process frames (default: false)
- `save_results`: Save results to disk (default: true)
- `debug_visualization`: Enable debug images (default: true)
- `processing_rate`: Processing rate in Hz (default: 1.0)

**Usage:**
```bash
ros2 run vision sam_vision_pipeline
```

---

## Data Flow

```
Camera Hardware (Webcam/RealSense)
         ↓
   CV Bridge (OpenCV)
         ↓
   Camera Service Node
         ↓
   ROS2 Topics (/camera/*)
         ↓
   SAM Vision Pipeline Node
         ↓
   Vision Services (/vision/*)
         ↓
   Results & Visualizations
```

---

## Service Dependencies

Services must be called in this order:

```
1. /vision/detect_objects
         ↓
2. /vision/classify_objects
         ↓
   ├─→ 3. /vision/get_positions
   ├─→ 4. /vision/generate_grasps
   └─→ 5. /vision/build_scene_graph

OR

/vision/process_scene (runs all steps)
```

---

## Launch Options

### Option 1: Manual Launch (Two Terminals)

Terminal 1 - Camera:
```bash
ros2 run vision camera_service
```

Terminal 2 - Pipeline:
```bash
ros2 run vision sam_vision_pipeline
```

### Option 2: Launch File (Single Command)

```bash
# Webcam
ros2 launch vision vision_with_camera.launch.py

# RealSense
ros2 launch vision vision_with_camera.launch.py camera_type:=realsense

# Test image
ros2 launch vision vision_with_camera.launch.py \
  camera_type:=file \
  image_file:=Final-proj/src/arrange.jpg
```

---

## Testing

### Full System Test

```bash
# Terminal 1: Launch system
ros2 launch vision vision_with_camera.launch.py \
  camera_type:=file \
  image_file:=Final-proj/src/arrange.jpg

# Terminal 2: Run tests
ros2 run vision integration_test
```

### Expected Output

```
==================================================================
Integration Test Summary
==================================================================
  [PASS] Reset Pipeline
  [PASS] Object Detection
  [PASS] Classification
  [PASS] Position Extraction
  [PASS] Grasp Generation
  [PASS] Scene Graph

Results: 6/6 tests passed
✓ All tests PASSED!
==================================================================
```

---

## Camera Service vs Vision Pipeline

| Aspect | Camera Service | Vision Pipeline |
|--------|---------------|-----------------|
| **Purpose** | Hardware interface | Image processing |
| **Uses CV Bridge** | ✅ Yes (OpenCV → ROS) | ✅ Yes (ROS → OpenCV) |
| **Opens Camera** | ✅ Yes | ❌ No |
| **AI Models** | ❌ No | ✅ Yes (SAM/CLIP/GraspNet) |
| **Publishes Images** | ✅ Yes | ❌ No (publishes results) |
| **Services** | 3 (start/stop/reset) | 7 (detect/classify/etc) |
| **Dependencies** | cv2, pyrealsense2 | torch, transformers |

---

## Key Points

1. **Camera Service** uses CV Bridge to open camera hardware and publish images
2. **Vision Pipeline** subscribes to camera topics (does NOT open camera)
3. Both nodes use CV Bridge (camera for OpenCV→ROS, pipeline for ROS→OpenCV)
4. Launch file starts both nodes together for complete system
5. Services can be called individually or via full pipeline

---

## Configuration Examples

### High Resolution RealSense
```bash
ros2 run vision camera_service --ros-args \
  -p camera_type:=realsense \
  -p width:=1280 \
  -p height:=720 \
  -p fps:=30.0
```

### Low Latency Webcam
```bash
ros2 run vision camera_service --ros-args \
  -p camera_type:=webcam \
  -p camera_id:=0 \
  -p width:=320 \
  -p height:=240 \
  -p fps:=60.0
```

### Video File Loop
```bash
ros2 run vision camera_service --ros-args \
  -p camera_type:=file \
  -p image_file:=test_video.mp4
```

---

For more details:
- **Service API:** See `docs/SERVICE_REFERENCE.md`
- **Testing:** See `docs/TESTING.md`
- **Quick Start:** See `docs/service_main.md`
