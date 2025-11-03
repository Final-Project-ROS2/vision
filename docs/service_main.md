# Vision Pipeline Services - Quick Start Guide

## Overview

The vision system provides **two workflows** for robotic vision:

1. **🖼️ Static Image Pipeline** - Process existing images from files
2. **📹 Webcam Pipeline** - Capture and process live webcam images

Both workflows use the same **4-stage vision pipeline**:
- **Stage 1:** SAM Object Detection & Segmentation
- **Stage 2:** CLIP Semantic Classification
- **Stage 3:** GraspNet 6D Grasp Pose Generation
- **Stage 4:** Scene Graph Construction & Spatial Relations

---

## System Architecture

### Workflow 1: Static Image Pipeline
```
┌─────────────────┐
│ External Node   │
│  Publishing to  │
│ /camera/        │
│   image_raw     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐      ┌──────────────────────┐
│ Camera Service  │─────→│  SAM Vision Pipeline │
│ (subscribe mode)│ ROS2 │  (4-stage process)   │
│                 │Topics│                      │
│ Subscribes &    │      │ Services:            │
│ Re-publishes to │      │  - detect_objects    │
│ /camera/        │      │  - classify_objects  │
│   image_raw     │      │  - generate_grasps   │
└─────────────────┘      │  - build_scene_graph │
                         └──────────────────────┘
```

### Workflow 2: Webcam Capture Pipeline
```
┌─────────────────┐
│    Webcam       │
│   /dev/video0   │
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────┐
│   Camera Service (Single Shot)  │
│   - Captures ONE image          │
│   - Saves to src-webcam/        │
│   - Auto-exits after capture    │
└────────┬────────────────────────┘
         │
         ↓
┌─────────────────┐      ┌──────────────────────┐
│ Camera Service  │─────→│  SAM Vision Pipeline │
│   (file mode)   │ ROS2 │  (4-stage process)   │
│ Loads saved img │Topics│                      │
└─────────────────┘      └──────────────────────┘
```

---

---

## 🖼️ Workflow 1: Static Image Pipeline

**Use Case:** Process images from an external camera node that publishes to `/camera/image_raw`.

### Step-by-Step Guide

#### Step 1: Start External Camera Node
First, ensure you have a node publishing images to `/camera/image_raw`. This could be:
- A ROS2 camera driver (e.g., `usb_cam`, `realsense2_camera`)
- A custom image publisher node
- A bag file playback: `ros2 bag play your_bag.db3`
- The `show_rgb_image` viewer node (displays and republishes images)

**Example A - Using RGB Image Viewer:**
```bash
# Terminal 1: Start RGB viewer (subscribes to /camera/image_raw)
ros2 run vision show_rgb_image
```

**Expected Output:**
```
[INFO] [1762167688.516158727] [rgb_image_viewer]: RGBImageViewer started.
[INFO] [1762167688.516303761] [rgb_image_viewer]: Services: /show_rgb_image, /toggle_continuous_display
[INFO] [1762167688.516400895] [rgb_image_viewer]: Subscribing to: /camera/image_raw
```

**Example B - Using Image Publisher:**
```bash
# Terminal 1: Publish a static image
ros2 run image_publisher image_publisher_node \
  /home/group11/final_project_ws/src/vision/test_images/my_scene.jpg
```

#### Step 2: Verify Image Topic
Check that images are being published:
```bash
ros2 topic echo /camera/image_raw --no-arr
```

You should see image messages flowing.

#### Step 3: Start Camera Service (Subscribe Mode)
**Terminal 2:**
```bash
cd /home/group11/final_project_ws
source install/local_setup.bash

ros2 run vision camera_service --ros-args \
  -p camera_type:=subscribe
```

**What this does:**
- Subscribes to `/camera/image_raw` topic
- Receives images from external node
- Converts ROS Image → OpenCV using CV Bridge
- Re-publishes or makes available for vision pipeline

#### Step 4: Start Vision Pipeline
**Terminal 3:**
```bash
cd /home/group11/final_project_ws
source install/local_setup.bash

ros2 run vision sam_vision_pipeline
```

**What this does:**
- Subscribes to camera topics
- Receives the image
- Waits for service calls to process it

#### Step 5: Process the Image

**Option A - Run Full Pipeline (Recommended):**
**Terminal 4:**
```bash
ros2 service call /vision/process_scene std_srvs/srv/Trigger
```

**Option B - Run Step-by-Step:**
**Terminal 4:**
```bash
# Step 1: Detect objects
ros2 service call /vision/detect_objects std_srvs/srv/Trigger

# Step 2: Classify detected objects
ros2 service call /vision/classify_objects std_srvs/srv/Trigger

# Step 3: Get 3D positions
ros2 service call /vision/get_positions std_srvs/srv/Trigger

# Step 4: Generate grasp poses
ros2 service call /vision/generate_grasps std_srvs/srv/Trigger

# Step 5: Build scene graph
ros2 service call /vision/build_scene_graph std_srvs/srv/Trigger
```

#### Step 6: View Results
Results are saved to:
```bash
~/ros2_vision_outputs/scene_YYYYMMDD_HHMMSS/
```

---

## 📹 Workflow 2: Webcam Capture Pipeline

**Use Case:** Capture a live image from webcam, save it, then process it.

### Step-by-Step Guide

#### Step 1: Capture Image from Webcam
**Terminal 1:**
```bash
cd /home/group11/final_project_ws
source install/local_setup.bash

# Capture ONE image and auto-exit
ros2 run vision camera_service --ros-args \
  -p save_images:=true \
  -p capture_single_shot:=true
```

**What this does:**
- Opens webcam at `/dev/video0`
- Captures ONE frame
- Saves to `/home/group11/final_project_ws/src/vision/src-webcam/webcam_TIMESTAMP.jpg`
- Automatically exits

**Output:**
```
[INFO] Webcam opened successfully
[INFO] 💾 Saved frame #1: webcam_20251103_174559_892545.jpg
[INFO] ✅ Single shot captured! Shutting down...
```

#### Step 2: Verify Image Was Captured
```bash
ls -lht /home/group11/final_project_ws/src/vision/src-webcam/*.jpg | head -1
```

You should see the most recent captured image.

#### Step 3: Start Camera Service with Captured Image
**Terminal 1:**
```bash
cd /home/group11/final_project_ws
source install/local_setup.bash

# Get the latest captured image filename
LATEST_IMAGE=$(ls -t /home/group11/final_project_ws/src/vision/src-webcam/*.jpg | head -1)

# Load it for processing
ros2 run vision camera_service --ros-args \
  -p camera_type:=file \
  -p image_file:=$LATEST_IMAGE
```

#### Step 4: Start Vision Pipeline
**Terminal 2:**
```bash
cd /home/group11/final_project_ws
source install/local_setup.bash

ros2 run vision sam_vision_pipeline
```

#### Step 5: Process the Captured Image
**Terminal 3:**
```bash
# Run full pipeline
ros2 service call /vision/process_scene std_srvs/srv/Trigger

# OR step-by-step (same as Workflow 1)
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
ros2 service call /vision/classify_objects std_srvs/srv/Trigger
ros2 service call /vision/get_positions std_srvs/srv/Trigger
ros2 service call /vision/generate_grasps std_srvs/srv/Trigger
ros2 service call /vision/build_scene_graph std_srvs/srv/Trigger
```

#### Step 6: View Results
Results are saved to:
```bash
~/ros2_vision_outputs/scene_YYYYMMDD_HHMMSS/
```

---

## 🚀 Quick Start Scripts

### Script 1: Process Image from Subscriber
```bash
#!/bin/bash
cd /home/group11/final_project_ws
source install/local_setup.bash

# Terminal 1 (run external camera publisher in background)
# Example: ros2 run image_publisher image_publisher_node /path/to/image.jpg &

# Terminal 2 (subscribe to /camera/image_raw)
ros2 run vision camera_service --ros-args \
  -p camera_type:=subscribe &

sleep 3

# Terminal 3 (run vision pipeline in background)
ros2 run vision sam_vision_pipeline &

sleep 5

# Terminal 4 (process)
ros2 service call /vision/process_scene std_srvs/srv/Trigger

echo "✅ Processing complete! Check ~/ros2_vision_outputs/ for results"
```

### Script 2: Capture and Process Webcam Image
```bash
#!/bin/bash
cd /home/group11/final_project_ws
source install/local_setup.bash

# Step 1: Capture from webcam
echo "📸 Capturing image from webcam..."
ros2 run vision camera_service --ros-args \
  -p save_images:=true \
  -p capture_single_shot:=true

# Step 2: Get latest image
LATEST_IMAGE=$(ls -t /home/group11/final_project_ws/src/vision/src-webcam/*.jpg | head -1)
echo "✅ Captured: $LATEST_IMAGE"

# Step 3: Process it
echo "🔄 Starting vision pipeline..."
ros2 run vision camera_service --ros-args \
  -p camera_type:=file \
  -p image_file:=$LATEST_IMAGE &

sleep 3

ros2 run vision sam_vision_pipeline &

sleep 5

ros2 service call /vision/process_scene std_srvs/srv/Trigger

echo "✅ Processing complete! Check ~/ros2_vision_outputs/ for results"
```

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

## 🔍 Real CLI Examples

### Example 1: Running RGB Image Viewer Node

The `show_rgb_image` node subscribes to `/camera/image_raw` and displays images:

```bash
$ ros2 run vision show_rgb_image
[INFO] [1762167688.516158727] [rgb_image_viewer]: RGBImageViewer started.
[INFO] [1762167688.516303761] [rgb_image_viewer]: Services: /show_rgb_image, /toggle_continuous_display
[INFO] [1762167688.516400895] [rgb_image_viewer]: Subscribing to: /camera/image_raw
# Node runs and displays images from /camera/image_raw
# Press Ctrl+C to stop
```

**What this node does:**
- Subscribes to `/camera/image_raw` topic
- Displays RGB images in OpenCV window
- Provides services to control display behavior
- Uses CV Bridge to convert ROS Image → OpenCV format

**Available Services:**
- `/show_rgb_image` - Display a single image
- `/toggle_continuous_display` - Toggle continuous display mode

### Example 2: Complete Workflow with Real Commands

```bash
# Terminal 1: Start webcam capture
$ ros2 run vision camera_service
[INFO] Webcam opened successfully
[INFO] Publishing to /camera/image_raw

# Terminal 2: View the camera feed (optional)
$ ros2 run vision show_rgb_image
[INFO] [rgb_image_viewer]: RGBImageViewer started.
[INFO] [rgb_image_viewer]: Subscribing to: /camera/image_raw

# Terminal 3: Start vision pipeline
$ ros2 run vision sam_vision_pipeline
[INFO] SAM Vision Pipeline started
[INFO] Subscribing to /camera/image_raw

# Terminal 4: Process the scene
$ ros2 service call /vision/process_scene std_srvs/srv/Trigger
waiting for service to become available...
requester: making request: std_srvs.srv.Trigger_Request()

response:
std_srvs.srv.Trigger_Response(success=True, message='Scene processed successfully')
```

### Example 3: Checking Active Topics

```bash
# List all camera topics
$ ros2 topic list | grep camera
/camera/camera_info
/camera/depth/image_raw
/camera/image_raw

# Check image messages
$ ros2 topic echo /camera/image_raw --no-arr
header:
  stamp:
    sec: 1762167688
    nanosec: 516158727
  frame_id: camera_frame
height: 480
width: 640
encoding: bgr8
is_bigendian: 0
step: 1920
# data: [omitted]

# Check topic frequency
$ ros2 topic hz /camera/image_raw
average rate: 30.002
  min: 0.033s max: 0.034s std dev: 0.00012s window: 30
```

---

For detailed documentation, see `SERVICE_REFERENCE.md` and `TESTING_GUIDE.md`.
