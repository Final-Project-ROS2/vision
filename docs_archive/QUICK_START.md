# Vision System Quick Reference

## 🎯 Quick Start

### Single Command (RECOMMENDED)
```bash
ros2 launch vision vision_with_camera.launch.py camera_type:=file image_file:=Final-proj/src/arrange.jpg
```

Then in another terminal:
```bash
ros2 service call /vision/process_scene std_srvs/srv/Trigger
```

---

## 📦 Two Main Nodes

### 1. Camera Service (Opens Camera via CV Bridge)
```bash
ros2 run vision camera_service
```

**What it does:** Opens webcam/depth camera hardware, publishes images

**Topics:** `/camera/image_raw`, `/camera/depth/image_raw`, `/camera/camera_info`

**Services:** `/camera/start`, `/camera/stop`, `/camera/reset`

### 2. SAM Vision Pipeline (Processes Images)
```bash
ros2 run vision sam_vision_pipeline
```

**What it does:** AI-powered object detection, classification, grasping

**Services:** 7 vision services (detect, classify, grasps, positions, etc.)

---

## 🎥 Camera Options

### Webcam
```bash
ros2 run vision camera_service
```

### Intel RealSense
```bash
ros2 run vision camera_service --ros-args -p camera_type:=realsense
```

### Test Image
```bash
ros2 run vision camera_service --ros-args -p camera_type:=file -p image_file:=Final-proj/src/arrange.jpg
```

---

## 🔧 Vision Services (7 Total)

| Service | Purpose | Dependencies |
|---------|---------|-------------|
| `/vision/detect_objects` | SAM object detection | Camera data |
| `/vision/classify_objects` | CLIP classification | detect_objects |
| `/vision/get_positions` | 3D positions | classify_objects |
| `/vision/generate_grasps` | 6D grasp poses | classify_objects |
| `/vision/build_scene_graph` | Scene understanding | classify_objects |
| `/vision/process_scene` | Full pipeline | Camera data |
| `/vision/reset_pipeline` | Clear cache | None |

---

## 🧪 Testing

### Automated Test
```bash
# Terminal 1
ros2 launch vision vision_with_camera.launch.py camera_type:=file image_file:=Final-proj/src/arrange.jpg

# Terminal 2
ros2 run vision integration_test
```

### Manual Test
```bash
# Terminal 1
ros2 run vision camera_service --ros-args -p camera_type:=file -p image_file:=Final-proj/src/arrange.jpg

# Terminal 2
ros2 run vision sam_vision_pipeline

# Terminal 3
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
ros2 service call /vision/classify_objects std_srvs/srv/Trigger
ros2 service call /vision/get_positions std_srvs/srv/Trigger
```

---

## 📊 Service Call Order

```
detect_objects → classify_objects → {positions, grasps, scene_graph}
```

OR just:
```
process_scene (runs all)
```

---

## 🔍 Checking Status

### Check nodes running
```bash
ros2 node list
```

Should show:
- `/camera_service`
- `/sam_vision_pipeline`

### Check topics
```bash
ros2 topic list
```

Should include:
- `/camera/image_raw`
- `/camera/camera_info`
- `/vision/debug_image`

### Check services
```bash
ros2 service list | grep -E "camera|vision"
```

Should show 10 services (3 camera + 7 vision)

---

## ⚡ Common Commands

### View camera stream
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
