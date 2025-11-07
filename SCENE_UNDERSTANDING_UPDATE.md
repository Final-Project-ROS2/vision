# Scene Understanding Service Update

## Changes Made

The `/vision/understand_scene` service has been updated to:

1. **Use `std_srvs/srv/Trigger` service type** instead of custom service type
2. **Support multiple calls** - can be called repeatedly to update scene analysis
3. **Provide JSON responses** with detailed scene information

## Usage

### Correct Service Call Format

```bash
ros2 service call /vision/understand_scene std_srvs/srv/Trigger
```

### Call Multiple Times

The service can be called as many times as needed. Each call will:
- Re-run object detection
- Re-compute spatial relationships
- Update the visualization
- Publish new results to `/vision/scene_understanding` topic

Example:
```bash
# First call
ros2 service call /vision/understand_scene std_srvs/srv/Trigger

# Wait a moment, then call again to update
ros2 service call /vision/understand_scene std_srvs/srv/Trigger

# Can be called as many times as needed
ros2 service call /vision/understand_scene std_srvs/srv/Trigger
```

### Test Script

A test script is provided to demonstrate multiple calls:

```bash
cd /home/group11/final_project_ws/src/vision
./testsh/test_understand_scene.sh
```

## Response Format

The service returns a JSON response with:

```json
{
  "success": true,
  "scene_id": "scene_20250107_123456",
  "total_objects": 5,
  "total_relations": 12,
  "graspable_objects": 3,
  "scene_description": "Scene contains 5 objects. 3 objects are graspable...",
  "timestamp": "2025-01-07T12:34:56Z"
}
```

## How It Works

Each time the service is called:

1. ✅ Calls `/vision/detect_objects` to get current detections
2. ✅ Calls `/vision/detect_grasp` to get grasp information
3. ✅ Computes spatial relationships between all object pairs
4. ✅ Generates natural language scene description
5. ✅ Publishes `SceneUnderstanding` message
6. ✅ Updates OpenCV visualization window
7. ✅ Saves visualization image to disk

## Prerequisites

Make sure these services are running:

```bash
# Terminal 1: SAM detector
ros2 run vision simple_sam_detector

# Terminal 2: CLIP classifier
ros2 run vision clip_classifier

# Terminal 3: GraspNet detector
ros2 run vision graspnet_detector

# Terminal 4: Scene understanding (this node)
ros2 run vision scene_understanding
```

## Key Changes in Code

### Service Creation
```python
# Now always uses Trigger service type
self.understand_service = self.create_service(
    Trigger,
    '/vision/understand_scene',
    self.understand_scene_callback,
    callback_group=self.callback_group
)
```

### Callback Function
- Updated to work with `Trigger` request/response format
- Returns JSON-formatted message with scene details
- Properly handles re-runs without state conflicts
- Better logging with visual separators

### Documentation
- All references updated to show `std_srvs/srv/Trigger`
- Usage instructions updated throughout the code
- OpenCV window now shows correct command

## Benefits

✅ **Standard Interface**: Uses ROS2 standard message type  
✅ **Reusable**: Can be called multiple times without issues  
✅ **Clear Responses**: JSON format provides structured data  
✅ **Better Logging**: Improved visual feedback in terminal  
✅ **Backward Compatible**: Works with existing pipeline nodes  

## Testing

To verify the service works correctly:

1. Start all prerequisite nodes
2. Call the service: `ros2 service call /vision/understand_scene std_srvs/srv/Trigger`
3. Check the response message (should show success: true)
4. Look at the OpenCV window for visualization
5. Call it again to verify it updates properly

The service can be called as many times as needed without restarting the node.
