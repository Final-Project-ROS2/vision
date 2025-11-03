# Vision Pipeline Services

## Overview

The vision system consists of two main components:
1. **Camera Service** - Handles camera hardware and publishes image streams
2. **SAM Vision Pipeline** - Processes images for object detection, classification, and scene understanding

---

## Camera Service

The Camera Service Node opens webcam/depth camera and publishes images via CV Bridge.

### Camera Topics Published

- `/camera/image_raw` - RGB images (sensor_msgs/Image)
- `/camera/depth/image_raw` - Depth images (sensor_msgs/Image)
- `/camera/camera_info` - Camera intrinsic parameters (sensor_msgs/CameraInfo)

### Camera Services

#### `/camera/start`
Start camera streaming

```bash
ros2 service call /camera/start std_srvs/srv/Trigger
```

#### `/camera/stop`
Stop camera streaming

```bash
ros2 service call /camera/stop std_srvs/srv/Trigger
```

#### `/camera/reset`
Reset camera connection

```bash
ros2 service call /camera/reset std_srvs/srv/Trigger
```

### Starting the Camera

**Option 1: Webcam (Default)**
```bash
ros2 run vision camera_service
```

**Option 2: Intel RealSense**
```bash
ros2 run vision camera_service --ros-args \
  -p camera_type:=realsense \
  -p width:=640 \
  -p height:=480 \
  -p fps:=30.0
```

**Option 3: Static Image File**
```bash
ros2 run vision camera_service --ros-args \
  -p camera_type:=file \
  -p image_file:=Final-proj/src/arrange.jpg
```

**Parameters:**
- `camera_id` (int): Camera device ID (default: 0)
- `camera_type` (string): 'webcam', 'realsense', or 'file' (default: 'webcam')
- `image_file` (string): Path to image/video file (for 'file' type)
- `width` (int): Image width in pixels (default: 640)
- `height` (int): Image height in pixels (default: 480)
- `fps` (double): Frame rate (default: 30.0)
- `auto_start` (bool): Auto-start streaming on launch (default: true)

---

## Available ROS2 Services

The SAM Vision Pipeline provides 7 ROS2 services for robotic vision tasks:

### 1. `/vision/detect_objects`
**Object Detection with SAM**

Detects and segments objects using Meta's Segment Anything Model.

```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

**Response:**
- Success: `Detected N objects`
- Failure: `No RGB image available`

---

### 2. `/vision/classify_objects`
**Semantic Classification with CLIP**

Classifies detected objects with semantic labels using OpenAI's CLIP model.

```bash
ros2 service call /vision/classify_objects std_srvs/srv/Trigger
```

**Prerequisites:** Must call `/vision/detect_objects` first

**Response:**
- Success: `Classified N objects`
- Failure: `No detections available`

---

### 3. `/vision/generate_grasps`
**6D Grasp Pose Generation**

Generates 6D grasp poses for manipulation using GraspNet approach.

```bash
ros2 service call /vision/generate_grasps std_srvs/srv/Trigger
```

**Prerequisites:** 
- Must call `/vision/classify_objects` first
- Requires depth data

**Response:**
- Success: `Generated N grasp poses`
- Publishes to: `/vision/grasp_poses`

---

### 4. `/vision/get_positions`
**3D Object Position Extraction**

Extracts 3D positions (x, y, z) for all detected objects using depth data.

```bash
ros2 service call /vision/get_positions std_srvs/srv/Trigger
```

**Prerequisites:** Must call `/vision/classify_objects` first

**Response:**
- Success: `Retrieved N object positions`

---

### 5. `/vision/build_scene_graph`
**Scene Understanding**

Constructs spatial relationships and scene graph between objects.

```bash
ros2 service call /vision/build_scene_graph std_srvs/srv/Trigger
```

**Prerequisites:** Must call `/vision/classify_objects` first

**Response:**
- Success: `Scene graph built with N objects and M relations`

---

### 6. `/vision/process_scene`
**Full Pipeline Execution**

Runs the complete 4-stage pipeline: Detection → Classification → Grasps → Scene Graph

```bash
ros2 service call /vision/process_scene std_srvs/srv/Trigger
```

**Prerequisites:** RGB and depth camera data available

**Response:**
- Success: `Scene processed successfully`

---

### 7. `/vision/reset_pipeline`
**Reset All State**

Clears all cached results and resets pipeline state.

```bash
ros2 service call /vision/reset_pipeline std_srvs/srv/Trigger
```

**Response:**
- Success: `Pipeline reset successfully`

---

## Usage Examples

### Step-by-Step Processing
```bash
# Step 1: Detect objects
ros2 service call /vision/detect_objects std_srvs/srv/Trigger

# Step 2: Classify detected objects
ros2 service call /vision/classify_objects std_srvs/srv/Trigger

# Step 3: Extract positions
ros2 service call /vision/get_positions std_srvs/srv/Trigger

# Step 4: Generate grasps
ros2 service call /vision/generate_grasps std_srvs/srv/Trigger

# Step 5: Build scene graph
ros2 service call /vision/build_scene_graph std_srvs/srv/Trigger
```

### Quick Full Pipeline
```bash
ros2 service call /vision/process_scene std_srvs/srv/Trigger
```

---

## Service Dependencies

```
Camera Data (RGB + Depth)
    ↓
/vision/detect_objects
    ↓
/vision/classify_objects
    ↓
    ├→ /vision/get_positions
    ├→ /vision/generate_grasps
    └→ /vision/build_scene_graph
```

---

## Complete System Usage

### Terminal 1: Start Camera Service
```bash
# For webcam
ros2 run vision camera_service

# OR for RealSense
ros2 run vision camera_service --ros-args -p camera_type:=realsense

# OR for static test image
ros2 run vision camera_service --ros-args -p camera_type:=file -p image_file:=Final-proj/src/arrange.jpg
```

### Terminal 2: Start Vision Pipeline
```bash
ros2 run vision sam_vision_pipeline
```

### Terminal 3: Call Services
```bash
# Test all services
ros2 run vision test_services

# OR test individual services
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
ros2 service call /vision/classify_objects std_srvs/srv/Trigger
ros2 service call /vision/get_positions std_srvs/srv/Trigger
```

---

## Testing Services

### Automated Integration Test
```bash
# Terminal 1: Camera
ros2 run vision camera_service --ros-args -p camera_type:=file -p image_file:=Final-proj/src/arrange.jpg

# Terminal 2: Pipeline
ros2 run vision sam_vision_pipeline

# Terminal 3: Test
ros2 run vision integration_test
```

### Manual Testing
```bash
# Start camera and pipeline as above, then:
ros2 run vision test_services
```

---

## Quick Start with Launch File

### Single Command Launch (RECOMMENDED)

Launch both camera and vision pipeline together:

```bash
# Webcam
ros2 launch vision vision_with_camera.launch.py

# RealSense depth camera
ros2 launch vision vision_with_camera.launch.py camera_type:=realsense

# Test with static image
ros2 launch vision vision_with_camera.launch.py \
  camera_type:=file \
  image_file:=Final-proj/src/arrange.jpg
```

Then call services:
```bash
ros2 service call /vision/process_scene std_srvs/srv/Trigger
```

---

## Topics Published

- `/camera/image_raw` - RGB images from camera
- `/camera/depth/image_raw` - Depth images (if available)
- `/camera/camera_info` - Camera intrinsic parameters
- `/vision/debug_image` - Visualization with detections and grasps
- `/vision/grasp_poses` - Generated grasp poses (geometry_msgs/PoseStamped)

---

For detailed documentation, see `SERVICE_REFERENCE.md` and `TESTING_GUIDE.md`.
