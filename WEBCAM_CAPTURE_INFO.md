# Webcam Image Capture Feature

## ✅ Implementation Complete!

The camera service now automatically saves webcam frames to disk when enabled.

---

## 📁 Where Webcam Images are Saved

**Location:**
```
/home/group11/final_project_ws/src/vision/src-webcam/
```

**Example saved images:**
```
webcam_20251103_173828_882260.jpg
webcam_20251103_173829_048922.jpg  
webcam_20251103_173829_215576.jpg
```

---

## 🚀 How to Use

### Start Camera Service with Image Saving

```bash
cd /home/group11/final_project_ws
source install/local_setup.bash

# Save every 30 frames (~1 second at 30fps)
ros2 run vision camera_service --ros-args \
  -p save_images:=true \
  -p save_interval:=30
```

### Parameters

- `save_images` (bool): Enable/disable saving (default: false)
- `save_interval` (int): Save every N frames (default: 30)
  - 5 = ~6 images per second
  - 10 = ~3 images per second  
  - 30 = ~1 image per second
  - 90 = ~1 image every 3 seconds

---

## 📸 Quick Test (3 Images)

```bash
# Start camera service with fast capture (every 5 frames)
cd /home/group11/final_project_ws
source install/local_setup.bash
ros2 run vision camera_service --ros-args -p save_images:=true -p save_interval:=5 &

# Wait a few seconds
sleep 3

# View saved images
ls -lht /home/group11/final_project_ws/src/vision/src-webcam/*.jpg | head -5

# Stop camera
pkill -f camera_service
```

---

## 🔄 How It Works

1. **Camera Service starts** and opens webcam
2. **Continuously streams** images to ROS2 topics (`/camera/image_raw`)
3. **Automatically saves** frames to `src-webcam/` based on `save_interval`
4. **Vision Pipeline** can process the live stream
5. **Saved images** remain on disk for later use

### File naming format:
```
webcam_YYYYMMDD_HHMMSS_microseconds.jpg
Example: webcam_20251103_173038_679282.jpg
         -------20251103-173038-679282---
         Year|Mo|Day|Hr|Mi|Sec|Microsec
```

---

## 📊 Integration with Vision Pipeline

The webcam images are:
- ✅ **Streamed live** to vision pipeline via ROS2 topics
- ✅ **Saved to disk** automatically for offline processing
- ✅ **Timestamped** for easy tracking
- ✅ **Available immediately** for detection services

### Full Pipeline Workflow:

```bash
# Terminal 1: Camera with saving
ros2 run vision camera_service --ros-args -p save_images:=true

# Terminal 2: Vision pipeline  
ros2 run vision sam_vision_pipeline

# Terminal 3: Detect objects from current webcam frame
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

---

## 📝 Notes

- Images are saved in **JPG format** (~60KB each)
- **No limit** on number of saved images (disk space permitting)
- Saving runs in **parallel** with ROS2 publishing (no performance impact)
- Images are saved **only when webcam is active** (not for file/realsense modes)

---

## 🔍 Verify Saved Images

```bash
# List all saved webcam images
ls -lh /home/group11/final_project_ws/src/vision/src-webcam/

# View most recent 3 images
ls -lht /home/group11/final_project_ws/src/vision/src-webcam/*.jpg | head -3

# Count total images
ls /home/group11/final_project_ws/src/vision/src-webcam/*.jpg | wc -l
```

---

## ✅ Summary

- **Location**: `/home/group11/final_project_ws/src/vision/src-webcam/`
- **Format**: `webcam_YYYYMMDD_HHMMSS_microseconds.jpg`
- **Enable**: `ros2 run vision camera_service --ros-args -p save_images:=true`
- **Control rate**: Use `-p save_interval:=N` parameter
- **Test**: Run for 3 seconds with `save_interval:=5` to get ~3 images
