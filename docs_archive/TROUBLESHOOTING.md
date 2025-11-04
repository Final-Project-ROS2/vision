# Troubleshooting Guide: SAM + CLIP Integration

## Issue: SAM Detection Timeout

### Symptoms
- `/vision/classify_detect` service times out
- Error: "SAM detection timeout after 30 seconds"
- No bounding boxes sent to CLIP

### Diagnosis Steps

**1. Run Health Check**
```bash
cd /home/group11/final_project_ws/src/vision
./check_pipeline_health.sh
```

**2. Check if SAM detector is running**
```bash
ros2 node list | grep simple_sam_detector
```

**3. Check if SAM service is available**
```bash
ros2 service list | grep detect_objects
```

**4. Test SAM detector directly**
```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

### Common Causes & Solutions

#### Cause 1: SAM Detector Not Running
**Symptom:** Service `/vision/detect_objects` not found

**Solution:**
```bash
# Start SAM detector
ros2 run vision simple_sam_detector
```

#### Cause 2: No Camera Feed
**Symptom:** SAM detector times out, no image received

**Check camera topic:**
```bash
# List topics
ros2 topic list | grep camera

# Check if publishing
ros2 topic hz /camera/image_raw
```

**Solution:** Start Gazebo or camera node
```bash
ros2 launch vision sam_gazebo_complete.launch.py
```

#### Cause 3: SAM Detector Stuck Processing
**Symptom:** Service call hangs, never returns

**Check detector logs:**
- Look for "Running SAM detection..." message
- Check if OpenCV window is frozen

**Solution:** Restart SAM detector
```bash
# Kill old process
pkill -f simple_sam_detector

# Restart
ros2 run vision simple_sam_detector
```

#### Cause 4: Service Call Blocking
**Symptom:** CLIP waits forever for SAM response

**Solution:** The updated code now uses:
- 30-second timeout (increased from 10s)
- Better async handling with `spin_once`
- Explicit error messages

#### Cause 5: No Objects Detected
**Symptom:** SAM returns empty detections list

**Check:**
- Is there anything in the camera view?
- Are objects contrasted from background?
- Check SAM detector OpenCV window

**Solution:**
- Adjust camera position/angle
- Add objects to scene
- Adjust SAM detection thresholds in `simple_sam_detector.py`

## Pipeline Execution Flow

### Expected Flow
```
1. User calls: /vision/classify_detect
   ↓
2. CLIP checks if /vision/detect_objects available (5s timeout)
   ↓
3. CLIP sends request to SAM detector
   ↓
4. SAM receives request, processes current image
   ↓ (max 30s)
5. SAM returns JSON with bboxes
   ↓
6. CLIP parses bboxes
   ↓
7. CLIP classifies each region
   ↓
8. CLIP returns combined results
```

### Timeout Values
- **Service availability check**: 5 seconds
- **SAM detection processing**: 30 seconds
- **Total pipeline**: ~31-35 seconds max

## Debugging Commands

### Check Service Communication
```bash
# Monitor service calls
ros2 service echo /vision/detect_objects

# Monitor topic flow
ros2 topic echo /camera/image_raw --once
```

### Check Node Health
```bash
# View CLIP classifier logs
ros2 node info /clip_classifier

# View SAM detector logs
ros2 node info /simple_sam_detector
```

### Manual Testing

**Test SAM only:**
```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

**Test CLIP only (full image):**
```bash
ros2 service call /vision/classify_all std_srvs/srv/Trigger
```

**Test integrated pipeline:**
```bash
ros2 service call /vision/classify_detect std_srvs/srv/Trigger
```

## Log Messages to Look For

### CLIP Classifier
**Healthy:**
```
✅ SAM detector service is ready!
🔍 Step 1/2: Calling /vision/detect_objects service...
✅ SAM detector service found, sending request...
⏳ Waiting for SAM detection to complete...
✅ SAM response received (success=True)
📋 Parsing SAM detection results...
✅ SAM data parsed successfully
📦 Found 1 detection sets
✅ SAM detected 3 objects: [[100, 150, 300, 400], ...]
🔍 Step 2/2: Classifying 3 detected regions with CLIP...
✅ Pipeline complete: 3 regions detected and classified
```

**Unhealthy:**
```
❌ /vision/detect_objects service not available
❌ SAM detection timeout after 30s
❌ Failed to parse SAM response
⚠️  No objects detected
```

### SAM Detector
**Healthy:**
```
🚀 Simple SAM Detector Started [CONTINUOUS MODE]
📡 Subscribing to: /camera/image_raw
🔍 Running SAM detection...
✅ Detection complete: 3 objects found
```

**Unhealthy:**
```
⚠️ No image available from /camera/image_raw
⚠️ No image received yet
```

## Performance Optimization

### If Pipeline is Slow

**1. Reduce Detection Area**
Modify `simple_sam_detector.py`:
```python
# Increase minimum area to filter small objects
min_area = (w * h) * 0.005  # Was 0.001
```

**2. Use GPU for CLIP**
Check CLIP is using CUDA:
```bash
# Look for "Device: cuda" in CLIP startup logs
ros2 run vision clip_classifier
```

**3. Reduce Candidate Labels**
Use fewer labels for faster classification:
```bash
ros2 run vision clip_classifier --labels "drill,gear,wrench"
```

## Quick Fix Checklist

- [ ] Gazebo/camera is running
- [ ] `/camera/image_raw` is publishing
- [ ] `simple_sam_detector` node is running
- [ ] `clip_classifier` node is running
- [ ] `/vision/detect_objects` service exists
- [ ] `/vision/classify_detect` service exists
- [ ] Objects are visible in camera view
- [ ] SAM OpenCV window shows green boxes
- [ ] CLIP OpenCV window is displaying

## Recovery Procedure

If pipeline is completely stuck:

```bash
# 1. Kill all nodes
pkill -f simple_sam_detector
pkill -f clip_classifier

# 2. Wait 2 seconds
sleep 2

# 3. Restart in order
# Terminal 1:
ros2 run vision simple_sam_detector

# Terminal 2 (wait for SAM to be ready):
ros2 run vision clip_classifier

# 4. Wait for "SAM detector service is ready!" message

# 5. Test
ros2 service call /vision/classify_detect std_srvs/srv/Trigger
```

## Advanced Debugging

### Enable Debug Logging
Modify `clip_classifier.py` temporarily:
```python
# In classify_detect_callback, add:
self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
```

### Trace Service Calls
```bash
# Terminal 1: Monitor SAM detector
ros2 run vision simple_sam_detector

# Terminal 2: Monitor service calls
ros2 service echo /vision/detect_objects

# Terminal 3: Call pipeline
ros2 service call /vision/classify_detect std_srvs/srv/Trigger
```

### Check ROS2 Communication
```bash
# Verify ROS2 daemon
ros2 daemon status

# Restart if needed
ros2 daemon stop
ros2 daemon start
```

## Contact & Support

If issues persist after following this guide:

1. Run health check with verbose output:
   ```bash
   ./check_pipeline_health.sh --verbose > health_report.txt
   ```

2. Capture logs from both nodes

3. Check OpenCV windows for visual feedback

4. Verify CUDA/GPU availability if using GPU mode
