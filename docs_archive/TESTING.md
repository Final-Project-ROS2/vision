# Testing the SAM Vision Pipeline

This guide covers how to test all services with the test image from `Final-proj/src/arrange.jpg`.

## Quick Test Overview

The vision pipeline has **7 services** that need testing:
1. `/vision/reset_pipeline` - Reset cached data
2. `/vision/detect_objects` - SAM object detection
3. `/vision/classify_objects` - CLIP classification
4. `/vision/get_positions` - 3D position extraction
5. `/vision/generate_grasps` - GraspNet grasp poses
6. `/vision/build_scene_graph` - Scene understanding
7. `/vision/process_scene` - Full pipeline

## Prerequisites

```bash
# Build the package
cd ~/ros2_ws
colcon build --packages-select vision
source install/setup.bash
```

## Test Method 1: Automated Integration Test (RECOMMENDED)

This runs all services in sequence with the test image.

```bash
# Terminal 1: Start the pipeline
ros2 run vision sam_vision_pipeline

# Terminal 2: Run integration test (wait 3 seconds after starting pipeline)
ros2 run vision integration_test
```

**Expected Output:**
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

## Test Method 2: Image-based Testing

Test with RGB/RGB-D images from disk.

```bash
# Terminal 1: Start pipeline
ros2 run vision sam_vision_pipeline

# Terminal 2: Publish test images
ros2 run vision test_pipeline_images --ros-args -p image_path:=Final-proj/src/arrange.jpg

# Terminal 3: Test services
ros2 run vision test_services
```

## Test Method 3: Manual Service Calls

Test individual services manually.

```bash
# Terminal 1: Start pipeline
ros2 run vision sam_vision_pipeline

# Terminal 2: Publish image (use quick script)
python scripts/quick_service_test.py

# Or manually:
ros2 topic pub /camera/image_raw sensor_msgs/msg/Image ...

# Terminal 3: Call services one by one
ros2 service call /vision/reset_pipeline std_srvs/srv/Trigger
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
ros2 service call /vision/classify_objects std_srvs/srv/Trigger
ros2 service call /vision/get_positions std_srvs/srv/Trigger
ros2 service call /vision/generate_grasps std_srvs/srv/Trigger
ros2 service call /vision/build_scene_graph std_srvs/srv/Trigger
```

## Service Dependencies

Services must be called in this order:

```
reset_pipeline (optional, clears cache)
    ↓
detect_objects (required first)
    ↓
classify_objects (required second)
    ↓
    ├─→ get_positions (needs classification)
    ├─→ generate_grasps (needs classification)
    └─→ build_scene_graph (needs classification)
```

## Troubleshooting

### Service Not Available
```bash
# Check if pipeline is running
ros2 node list | grep vision

# Check services
ros2 service list | grep vision
```

### No Detections
- Check if image is being published: `ros2 topic echo /camera/image_raw --once`
- Check image quality (not too dark/blurry)
- Try resetting pipeline: `ros2 service call /vision/reset_pipeline std_srvs/srv/Trigger`

### Position Service Fails
- Ensure depth data is available (RGB-D image or synthetic depth)
- Check that detection and classification ran successfully first
- Verify depth values are reasonable (not all zeros)

### All Services Timeout
- Increase timeout in test scripts (default 15 seconds)
- Check GPU availability for SAM/CLIP models
- Monitor system resources (RAM/VRAM usage)

## Test Image Location

The default test image is: `Final-proj/src/arrange.jpg`

This is an RGB image showing objects on a table. For full 3D position testing, depth data is synthesized automatically.

## Expected Results

### Detection
- Should find multiple objects (depends on image content)
- Each detection has: bbox, mask, confidence score

### Classification
- Each detected object gets semantic tags
- Tags include: object category, attributes, relationships

### Positions
- 3D coordinates for each object: {x, y, z}
- Coordinates in meters from camera frame
- Includes 2D bounding box and confidence

### Grasps
- 6-DoF grasp poses for each object
- Position (x, y, z) and orientation (quaternion)
- Grasp quality score

### Scene Graph
- Nodes: detected objects with attributes
- Edges: spatial relationships (on, next_to, etc.)
- JSON format

## Performance Benchmarks

On typical hardware (RTX 3080, 32GB RAM):
- Detection: 2-5 seconds
- Classification: 1-2 seconds
- Positions: <0.5 seconds
- Grasps: 3-8 seconds
- Scene Graph: <1 second

Total pipeline: 7-17 seconds per frame

## Next Steps

After confirming services work:
1. Test with real RGB-D camera in Gazebo
2. Integrate with manipulation pipeline
3. Test with robot arm for grasping
4. Optimize performance (caching, GPU batching)

## Scripts Location

All test scripts are in `scripts/`:
- `integration_test.py` - Full automated test
- `test_services.py` - Service sequence test
- `test_pipeline_images.py` - Image-based test
- `quick_service_test.py` - Manual test helper

## Documentation

- `SERVICE_REFERENCE.md` - Complete API docs
- `SAM_PIPELINE_SUMMARY.md` - Implementation details
- `QUICK_REFERENCE.md` - Command reference
