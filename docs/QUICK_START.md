# Vision Pipeline - Quick Start Guide# Vision System Quick Reference



## Overview## 🎯 Quick Start



The vision system provides a complete SAM + CLIP + GraspNet pipeline for robotic object detection, classification, and grasp generation.### Single Command (RECOMMENDED)

```bash

**Pipeline Stages:**ros2 launch vision vision_with_camera.launch.py camera_type:=file image_file:=Final-proj/src/arrange.jpg

1. **SAM** - Object Detection & Segmentation```

2. **CLIP** - Semantic Classification

3. **GraspNet** - 6D Grasp Pose GenerationThen in another terminal:

4. **Scene Graph** - Spatial Relations```bash

ros2 service call /vision/process_scene std_srvs/srv/Trigger

---```



## Installation---



```bash## 📦 Two Main Nodes

cd ~/final_project_ws/src/vision

bash install_pipeline.sh### 1. Camera Service (Opens Camera via CV Bridge)

``````bash

ros2 run vision camera_service

---```



## Quick Usage**What it does:** Opens webcam/depth camera hardware, publishes images



### Start the Vision Pipeline**Topics:** `/camera/image_raw`, `/camera/depth/image_raw`, `/camera/camera_info`



```bash**Services:** `/camera/start`, `/camera/stop`, `/camera/reset`

cd ~/final_project_ws

source install/setup.bash### 2. SAM Vision Pipeline (Processes Images)

ros2 launch vision vision.launch.py```bash

```ros2 run vision sam_vision_pipeline

```

### Test with Sample Image

**What it does:** AI-powered object detection, classification, grasping

**Terminal 1:** Start pipeline (from above)

**Services:** 7 vision services (detect, classify, grasps, positions, etc.)

**Terminal 2:** Run test

```bash---

cd ~/final_project_ws/src/vision

python3 vision_scripts/test_services.py## 🎥 Camera Options

```

### Webcam

---```bash

ros2 run vision camera_service

## Common Workflows```



### Workflow 1: Process Static Image### Intel RealSense

```bash

```bashros2 run vision camera_service --ros-args -p camera_type:=realsense

# Terminal 1: Start pipeline```

ros2 launch vision vision.launch.py

### Test Image

# Terminal 2: Call detection service```bash

ros2 service call /detect_objects vision_msgs/srv/DetectObjects \ros2 run vision camera_service --ros-args -p camera_type:=file -p image_file:=Final-proj/src/arrange.jpg

  "{image_path: '/path/to/image.jpg'}"```



# Get classifications, grasps, scene graph with additional service calls---

```

## 🔧 Vision Services (7 Total)

### Workflow 2: Webcam Capture & Process

| Service | Purpose | Dependencies |

```bash|---------|---------|-------------|

# Step 1: Capture image from webcam| `/vision/detect_objects` | SAM object detection | Camera data |

ros2 run vision camera_service_node --ros-args -p mode:=webcam -p capture_single:=true| `/vision/classify_objects` | CLIP classification | detect_objects |

| `/vision/get_positions` | 3D positions | classify_objects |

# Step 2: Process the captured image| `/vision/generate_grasps` | 6D grasp poses | classify_objects |

cd ~/final_project_ws/src/vision| `/vision/build_scene_graph` | Scene understanding | classify_objects |

python3 vision_scripts/test_services.py| `/vision/process_scene` | Full pipeline | Camera data |

```| `/vision/reset_pipeline` | Clear cache | None |



### Workflow 3: Subscribe to Camera Topic---



```bash## 🧪 Testing

# Terminal 1: Start external camera publisher

ros2 run your_camera_package camera_publisher### Automated Test

```bash

# Terminal 2: Start vision pipeline in subscribe mode# Terminal 1

ros2 run vision camera_service_node --ros-args -p mode:=subscriberos2 launch vision vision_with_camera.launch.py camera_type:=file image_file:=Final-proj/src/arrange.jpg



# Terminal 3: Call vision services# Terminal 2

python3 vision_scripts/test_services.pyros2 run vision integration_test

``````



---### Manual Test

```bash

## Available Services# Terminal 1

ros2 run vision camera_service --ros-args -p camera_type:=file -p image_file:=Final-proj/src/arrange.jpg

| Service | Description |

|---------|-------------|# Terminal 2

| `/detect_objects` | SAM object detection & segmentation |ros2 run vision sam_vision_pipeline

| `/classify_objects` | CLIP semantic classification |

| `/generate_grasps` | GraspNet 6D grasp poses |# Terminal 3

| `/build_scene_graph` | Spatial relationships |ros2 service call /vision/detect_objects std_srvs/srv/Trigger

ros2 service call /vision/classify_objects std_srvs/srv/Trigger

---ros2 service call /vision/get_positions std_srvs/srv/Trigger

```

## Troubleshooting

---

### Pipeline won't start

```bash## 📊 Service Call Order

# Check dependencies

bash check_pipeline_health.sh```

detect_objects → classify_objects → {positions, grasps, scene_graph}

# Rebuild```

cd ~/final_project_ws

colcon build --packages-select vision --symlink-installOR just:

source install/setup.bash```

```process_scene (runs all)

```

### Model download issues

```bash---

# Models auto-download on first run

# If stuck, manually download to ~/.cache/torch/hub/## 🔍 Checking Status

```

### Check nodes running

### Camera not found```bash

```bashros2 node list

# Check available cameras```

ls /dev/video*

Should show:

# Test webcam- `/camera_service`

ros2 run vision camera_service_node --ros-args -p mode:=webcam- `/sam_vision_pipeline`

```

### Check topics

---```bash

ros2 topic list

## Testing```



```bashShould include:

# Run all tests- `/camera/image_raw`

cd ~/final_project_ws/src/vision- `/camera/camera_info`

python3 vision_scripts/integration_test.py- `/vision/debug_image`



# Test individual components### Check services

bash testsh/test_integrated_pipeline.sh```bash

bash testsh/test_clip_classifier.shros2 service list | grep -E "camera|vision"

``````



---Should show 10 services (3 camera + 7 vision)



## Next Steps---



- See `ARCHITECTURE.md` for system design details## ⚡ Common Commands

- See `README.md` in project root for package information

- Check `vision_scripts/` for more example scripts### View camera stream

```bash
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

### View debug visualization
```bash
ros2 run rqt_image_view rqt_image_view /vision/debug_image
```

### Echo camera info
```bash
ros2 topic echo /camera/camera_info --once
```

### Check service response
```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

---

## 🐛 Troubleshooting

### Camera not working
```bash
# Check if camera service is running
ros2 node info /camera_service

# Try resetting camera
ros2 service call /camera/reset std_srvs/srv/Trigger

# Check if images are publishing
ros2 topic hz /camera/image_raw
```

### Pipeline not working
```bash
# Check if receiving images
ros2 topic echo /camera/image_raw --once

# Reset pipeline
ros2 service call /vision/reset_pipeline std_srvs/srv/Trigger

# Check logs
ros2 node info /sam_vision_pipeline
```

### Service call fails
```bash
# Make sure to call in order:
# 1. detect_objects (first)
# 2. classify_objects (second)
# 3. get_positions/grasps/scene_graph (after classify)
```

---

## 📚 Documentation

- **This file:** Quick reference
- **service_main.md:** Complete service documentation
- **ARCHITECTURE.md:** System architecture details
- **TESTING.md:** Comprehensive testing guide
- **SERVICE_REFERENCE.md:** Full API reference

---

## 💡 Key Points

1. ✅ **Camera Service** opens camera via CV Bridge (cv2.VideoCapture)
2. ✅ **Vision Pipeline** subscribes to camera topics (does NOT open camera)
3. ✅ Both use CV Bridge (camera: OpenCV→ROS, pipeline: ROS→OpenCV)
4. ✅ Launch file starts both nodes together
5. ✅ 7 modular services for flexible vision processing
6. ✅ Test image available at `Final-proj/src/arrange.jpg`

---

Built with ❤️ for ROS2 Humble
