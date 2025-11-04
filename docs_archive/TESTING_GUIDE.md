# Testing the Vision Pipeline

This guide covers how to test the SAM vision pipeline with RGB and RGB-D images.

## Test Overview

The vision pipeline can be tested with:
1. **Service-only testing** - Test each service independently
2. **Image-based testing** - Test with actual RGB/RGB-D images from disk
3. **Gazebo simulation** - Test with simulated camera in Gazebo

---

## 1. Service Testing

Test all ROS2 services to ensure they're properly implemented.

### Start the Pipeline

```bash
# Terminal 1: Start the vision pipeline
ros2 run vision sam_vision_pipeline
```

### Run Service Tests

```bash
# Terminal 2: Run automated service tests
ros2 run vision test_services
```

**Expected Output:**
```
Test Summary
======================================================================
  [PASS] reset
  [PASS] detection
  [PASS] classification
  [PASS] positions
  [PASS] grasps
  [PASS] scene_graph
  [PASS] full_pipeline

Results: 7/7 tests passed
```

### Manual Service Testing

```bash
# Test individual services
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
ros2 service call /vision/classify_objects std_srvs/srv/Trigger
ros2 service call /vision/get_positions std_srvs/srv/Trigger
ros2 service call /vision/generate_grasps std_srvs/srv/Trigger
ros2 service call /vision/build_scene_graph std_srvs/srv/Trigger
```

---

## 2. RGB Image Testing

Test the pipeline with RGB images from the src/pipeline directory.

### Test with Default Image

```bash
# Terminal 1: Start pipeline
ros2 run vision sam_vision_pipeline

# Terminal 2: Test with auto-discovered image
ros2 run vision test_pipeline_images
```

### Test with Specific Image

```bash
# Test with custom RGB image
ros2 run vision test_pipeline_images --ros-args \
    -p image_path:=/path/to/your/image.jpg \
    -p test_mode:=rgb \
    -p auto_test:=true
```

### Test with Image from Final-proj/src

```bash
# Use the test image from src directory
ros2 run vision test_pipeline_images --ros-args \
    -p image_path:=Final-proj/src/arrange.jpg \
    -p test_mode:=rgb
```

**Expected Output:**
```
======================================================================
Starting Automated Test Sequence
======================================================================

Testing: Detection
  [PASS] Detected 3 objects

Testing: Classification
  [PASS] Classified 3 objects

Testing: Position Extraction
  [PASS] Retrieved 3 object positions

Results: 3/3 tests passed
```

---

## 3. RGB-D Image Testing

Test with both RGB and depth images.

### Test with RGB + Depth

```bash
ros2 run vision test_pipeline_images --ros-args \
    -p image_path:=/path/to/rgb.jpg \
    -p depth_path:=/path/to/depth.png \
    -p test_mode:=rgbd \
    -p auto_test:=true
```

### Test with Synthetic Depth

If you only have RGB, the tester can create synthetic depth:

```bash
ros2 run vision test_pipeline_images --ros-args \
    -p image_path:=Final-proj/src/arrange.jpg \
    -p test_mode:=rgbd \
    -p auto_test:=true
```

**Expected Output:**
```
======================================================================
Starting Automated Test Sequence
======================================================================

Testing: Detection
  [PASS] Detected 3 objects

Testing: Classification
  [PASS] Classified 3 objects

Testing: Position Extraction
  [PASS] Retrieved 3 object positions

Testing: Grasp Generation
  [PASS] Generated 6 grasp poses

Testing: Scene Graph
  [PASS] Scene graph built with 3 objects and 5 relations

Results: 5/5 tests passed
```

---

## 4. Gazebo Simulation Testing

Test with simulated RGB-D camera in Gazebo.

### Start Gazebo with Camera

```bash
ros2 launch vision sam_gazebo_complete.launch.py
```

This automatically:
- Starts Gazebo with RGB-D camera
- Launches the vision pipeline node
- Begins publishing camera data

### Monitor Topics

```bash
# Check camera topics
ros2 topic list | grep camera

# View camera images
ros2 run rqt_image_view rqt_image_view /camera/image_raw
ros2 run rqt_image_view rqt_image_view /camera/depth/image_raw

# View processed results
ros2 run rqt_image_view rqt_image_view /vision/debug_image
```

### Test Services in Gazebo

```bash
# Process the current scene
ros2 service call /vision/process_scene std_srvs/srv/Trigger

# Check grasp poses
ros2 topic echo /vision/grasp_poses --once
```

---

## 5. Checking Test Results

### View Debug Visualizations

All test runs save visualization images:

```bash
# Service tests don't save images (services only)

# Image tests save to:
~/ros2_vision_outputs/scene_TIMESTAMP/

# Gazebo tests save to:
~/ros2_vision_outputs/scene_TIMESTAMP/
```

### Check Logs

```bash
# View node logs
ros2 run vision sam_vision_pipeline

# In another terminal, check topics
ros2 topic list

# Check service list
ros2 service list | grep vision
```

---

## 6. Using Test Images from src/pipeline

The main test image is located at:
```
Final-proj/src/arrange.jpg
```

### Quick Test with src Image

```bash
# Method 1: Let the tester find it automatically
ros2 run vision test_pipeline_images

# Method 2: Specify explicitly
ros2 run vision test_pipeline_images --ros-args \
    -p image_path:=$(pwd)/Final-proj/src/arrange.jpg
```

### Add Your Own Test Images

1. Place images in `Final-proj/data/test_images/`
2. Run with:
```bash
ros2 run vision test_pipeline_images --ros-args \
    -p image_path:=Final-proj/data/test_images/your_image.jpg
```

---

## 7. Common Test Scenarios

### Scenario 1: Quick Validation

Just verify everything works:
```bash
ros2 run vision sam_vision_pipeline &
sleep 3
ros2 run vision test_services
```

### Scenario 2: RGB-Only Workflow

Test detection and classification only:
```bash
ros2 run vision sam_vision_pipeline &
sleep 3
ros2 run vision test_pipeline_images --ros-args -p test_mode:=rgb
```

### Scenario 3: Full RGB-D Pipeline

Test complete pipeline with grasps:
```bash
ros2 run vision sam_vision_pipeline &
sleep 3
ros2 run vision test_pipeline_images --ros-args -p test_mode:=rgbd
```

### Scenario 4: Gazebo Integration

Test with simulated environment:
```bash
ros2 launch vision sam_gazebo_complete.launch.py
# Wait 5 seconds for everything to start
ros2 service call /vision/process_scene std_srvs/srv/Trigger
```

---

## 8. Troubleshooting

### Services Not Available

```bash
$ ros2 run vision test_services
ERROR: Not all services available

# Solution: Start the pipeline first
ros2 run vision sam_vision_pipeline
```

### No RGB Image Available

```bash
$ ros2 service call /vision/detect_objects std_srvs/srv/Trigger
response: std_srvs.srv.Trigger_Response(success=False, message='No RGB image available')

# Solution: Publish images or start Gazebo
ros2 run vision test_pipeline_images
```

### Test Image Not Found

```bash
$ ros2 run vision test_pipeline_images
WARN: No test image found

# Solution: Specify image path
ros2 run vision test_pipeline_images --ros-args \
    -p image_path:=Final-proj/src/arrange.jpg
```

---

## 9. Performance Benchmarking

Monitor processing times:

```bash
# Run pipeline with timing logs
ros2 run vision sam_vision_pipeline --ros-args --log-level debug

# Check processing rates
ros2 topic hz /vision/debug_image
ros2 topic hz /vision/grasp_poses
```

---

## 10. Advanced Testing

### Test All Services in Sequence

```python
#!/usr/bin/env python3
import subprocess
import time

services = [
    '/vision/detect_objects',
    '/vision/classify_objects',
    '/vision/get_positions',
    '/vision/generate_grasps',
    '/vision/build_scene_graph'
]

for service in services:
    print(f"Testing {service}...")
    subprocess.run(['ros2', 'service', 'call', service, 'std_srvs/srv/Trigger'])
    time.sleep(1)
```

### Continuous Testing

```bash
# Run tests in loop
while true; do
    ros2 service call /vision/process_scene std_srvs/srv/Trigger
    sleep 5
done
```

---

## Summary

| Test Type | Command | Purpose |
|-----------|---------|---------|
| Service Test | `ros2 run vision test_services` | Verify all services work |
| RGB Test | `ros2 run vision test_pipeline_images` | Test with RGB images |
| RGB-D Test | `ros2 run vision test_pipeline_images --ros-args -p test_mode:=rgbd` | Test with depth |
| Gazebo Test | `ros2 launch vision sam_gazebo_complete.launch.py` | Full simulation |

For detailed service documentation, see `SERVICE_REFERENCE.md`.
