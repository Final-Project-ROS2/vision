# SAM Vision Pipeline Service Reference

Complete reference for all ROS2 services provided by the SAM Vision Pipeline.

## Service Overview

The SAM Vision Pipeline provides 7 ROS2 services for modular access to the vision system:

| Service Name | Type | Description |
|-------------|------|-------------|
| `/vision/detect_objects` | Trigger | Run SAM object detection |
| `/vision/classify_objects` | Trigger | Run CLIP semantic classification |
| `/vision/generate_grasps` | Trigger | Generate 6D grasp poses |
| `/vision/get_positions` | Trigger | Extract 3D object positions |
| `/vision/build_scene_graph` | Trigger | Build scene graph with relations |
| `/vision/process_scene` | Trigger | Run complete pipeline |
| `/vision/reset_pipeline` | Trigger | Reset pipeline state |

## Service Dependencies

```
[Camera Data] → detect_objects → classify_objects → {generate_grasps, get_positions}
                                                   ↓
                                        build_scene_graph
```

## Detailed Service Documentation

### 1. Object Detection

**Service:** `/vision/detect_objects`

**Purpose:** Run SAM (Segment Anything Model) to detect and segment objects in the scene.

**Call:**
```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

**Response:**
- Success: `{success: True, message: "Detected N objects"}`
- Failure: `{success: False, message: "No RGB image available"}`

**Cached Results:** Stores detections and masks for use by downstream services.

**Example:**
```bash
$ ros2 service call /vision/detect_objects std_srvs/srv/Trigger
waiting for service to become available...
requester: making request: std_srvs.srv.Trigger_Request()

response:
std_srvs.srv.Trigger_Response(success=True, message='Detected 3 objects')
```

---

### 2. Object Classification

**Service:** `/vision/classify_objects`

**Purpose:** Use CLIP to add semantic labels and attributes to detected objects.

**Prerequisites:** Must call `/vision/detect_objects` first

**Call:**
```bash
ros2 service call /vision/classify_objects std_srvs/srv/Trigger
```

**Response:**
- Success: `{success: True, message: "Classified N objects"}`
- Failure: `{success: False, message: "No detections available. Run /vision/detect_objects first"}`

**Cached Results:** Stores semantic objects with labels, colors, materials, and affordances.

**Example:**
```bash
$ ros2 service call /vision/classify_objects std_srvs/srv/Trigger
response:
std_srvs.srv.Trigger_Response(success=True, message='Classified 3 objects')
```

---

### 3. Grasp Pose Generation

**Service:** `/vision/generate_grasps`

**Purpose:** Generate 6D grasp poses for all classified objects using GraspNet approach.

**Prerequisites:** 
- Must call `/vision/classify_objects` first
- Requires depth data

**Call:**
```bash
ros2 service call /vision/generate_grasps std_srvs/srv/Trigger
```

**Response:**
- Success: `{success: True, message: "Generated N grasp poses"}`
- Failure: `{success: False, message: "No classifications available..."}` or `{success: False, message: "No depth data available"}`

**Published Topics:** Publishes grasp poses to `/vision/grasp_poses` (geometry_msgs/PoseStamped)

**Cached Results:** Stores grasp candidates with quality scores, approach directions, and collision status.

**Example:**
```bash
$ ros2 service call /vision/generate_grasps std_srvs/srv/Trigger
response:
std_srvs.srv.Trigger_Response(success=True, message='Generated 6 grasp poses')
```

---

### 4. Position Extraction

**Service:** `/vision/get_positions`

**Purpose:** Extract 3D positions (x, y, z) for all classified objects using depth data.

**Prerequisites:** Must call `/vision/classify_objects` first

**Call:**
```bash
ros2 service call /vision/get_positions std_srvs/srv/Trigger
```

**Response:**
- Success: `{success: True, message: "Retrieved N object positions"}`
- Failure: `{success: False, message: "No classifications available..."}` 

**Cached Results:** Stores position data including:
- Object ID and class
- 3D position (x, y, z in meters)
- 2D bounding box
- Detection confidence

**Example:**
```bash
$ ros2 service call /vision/get_positions std_srvs/srv/Trigger
response:
std_srvs.srv.Trigger_Response(success=True, message='Retrieved 3 object positions')
```

---

### 5. Scene Graph Construction

**Service:** `/vision/build_scene_graph`

**Purpose:** Build a scene graph with spatial relationships between objects.

**Prerequisites:** Must call `/vision/classify_objects` first (optionally `/vision/generate_grasps`)

**Call:**
```bash
ros2 service call /vision/build_scene_graph std_srvs/srv/Trigger
```

**Response:**
- Success: `{success: True, message: "Scene graph built with N objects and M relations"}`
- Failure: `{success: False, message: "No classifications available..."}` 

**Cached Results:** Stores scene graph including:
- All objects with properties
- Spatial relations (near, above, below, left_of, right_of)
- Scene metadata (type, affordances, confidence)

**Example:**
```bash
$ ros2 service call /vision/build_scene_graph std_srvs/srv/Trigger
response:
std_srvs.srv.Trigger_Response(success=True, message='Scene graph built with 3 objects and 5 relations')
```

---

### 6. Full Pipeline Processing

**Service:** `/vision/process_scene`

**Purpose:** Run the complete 4-stage pipeline: Detection → Classification → Grasps → Scene Graph

**Prerequisites:** RGB and depth camera data available

**Call:**
```bash
ros2 service call /vision/process_scene std_srvs/srv/Trigger
```

**Response:**
- Success: `{success: True, message: "Scene processed successfully"}`
- Failure: `{success: False, message: "No RGB-D data available"}` or other error

**Published Topics:** 
- `/vision/debug_image` - Visualization with detections and grasps
- `/vision/grasp_poses` - All generated grasp poses

**Cached Results:** Updates all cached data (detections, classifications, grasps, positions, scene graph)

**Example:**
```bash
$ ros2 service call /vision/process_scene std_srvs/srv/Trigger
response:
std_srvs.srv.Trigger_Response(success=True, message='Scene processed successfully')
```

---

### 7. Pipeline Reset

**Service:** `/vision/reset_pipeline`

**Purpose:** Reset all pipeline state and clear cached results.

**Call:**
```bash
ros2 service call /vision/reset_pipeline std_srvs/srv/Trigger
```

**Response:**
- Success: `{success: True, message: "Pipeline reset successfully"}`

**Effects:**
- Clears all cached detections, classifications, grasps, positions, and scene graph
- Resets camera data
- Reinitializes pipeline if needed

**Example:**
```bash
$ ros2 service call /vision/reset_pipeline std_srvs/srv/Trigger
response:
std_srvs.srv.Trigger_Response(success=True, message='Pipeline reset successfully')
```

---

## Usage Patterns

### Pattern 1: Step-by-Step Processing

Process the scene in individual stages for fine-grained control:

```bash
# Step 1: Detect objects
ros2 service call /vision/detect_objects std_srvs/srv/Trigger

# Step 2: Classify detected objects
ros2 service call /vision/classify_objects std_srvs/srv/Trigger

# Step 3: Extract positions
ros2 service call /vision/get_positions std_srvs/srv/Trigger

# Step 4: Generate grasps
ros2 service call /vision/generate_grasps std_srvs/srv/Trigger

# Step 5: Build scene understanding
ros2 service call /vision/build_scene_graph std_srvs/srv/Trigger
```

### Pattern 2: Quick Full Pipeline

Process everything in one call:

```bash
ros2 service call /vision/process_scene std_srvs/srv/Trigger
```

### Pattern 3: Detection + Classification Only

For applications that only need object identification:

```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
ros2 service call /vision/classify_objects std_srvs/srv/Trigger
```

### Pattern 4: Grasping Focus

For manipulation applications:

```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
ros2 service call /vision/classify_objects std_srvs/srv/Trigger
ros2 service call /vision/generate_grasps std_srvs/srv/Trigger
```

---

## Testing Services

Use the provided test script to verify all services:

```bash
# Build the package
cd ~/ros2_ws
colcon build --packages-select vision

# Source the workspace
source install/setup.bash

# Start the vision pipeline
ros2 run vision sam_vision_pipeline

# In another terminal, run the service test
ros2 run vision test_services
```

---

## Monitoring Services

Check service availability:

```bash
# List all vision services
ros2 service list | grep vision

# Get service type
ros2 service type /vision/detect_objects

# View service details
ros2 service info /vision/detect_objects
```

---

## Integration Examples

### Python Client Example

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class VisionClient(Node):
    def __init__(self):
        super().__init__('vision_client')
        self.detect_client = self.create_client(Trigger, '/vision/detect_objects')
        self.classify_client = self.create_client(Trigger, '/vision/classify_objects')
    
    def process_scene(self):
        # Detect objects
        detect_req = Trigger.Request()
        detect_future = self.detect_client.call_async(detect_req)
        rclpy.spin_until_future_complete(self, detect_future)
        
        if detect_future.result().success:
            print(f"Detection: {detect_future.result().message}")
            
            # Classify objects
            classify_req = Trigger.Request()
            classify_future = self.classify_client.call_async(classify_req)
            rclpy.spin_until_future_complete(self, classify_future)
            
            if classify_future.result().success:
                print(f"Classification: {classify_future.result().message}")

def main():
    rclpy.init()
    client = VisionClient()
    client.process_scene()
    rclpy.shutdown()
```

### C++ Client Example

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

class VisionClient : public rclcpp::Node {
public:
    VisionClient() : Node("vision_client") {
        detect_client_ = create_client<std_srvs::srv::Trigger>("/vision/detect_objects");
        classify_client_ = create_client<std_srvs::srv::Trigger>("/vision/classify_objects");
    }
    
    void process_scene() {
        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
        
        // Detect objects
        auto future = detect_client_->async_send_request(request);
        if (rclcpp::spin_until_future_complete(shared_from_this(), future) ==
            rclcpp::FutureReturnCode::SUCCESS) {
            RCLCPP_INFO(get_logger(), "Detection: %s", future.get()->message.c_str());
        }
    }
    
private:
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr detect_client_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr classify_client_;
};
```

---

## Troubleshooting

### Service Not Available

```bash
$ ros2 service call /vision/detect_objects std_srvs/srv/Trigger
waiting for service to become available...
```

**Solution:** Start the vision pipeline node:
```bash
ros2 run vision sam_vision_pipeline
# or
ros2 launch vision sam_gazebo_complete.launch.py
```

### No RGB-D Data Available

```bash
response: std_srvs.srv.Trigger_Response(success=False, message='No RGB image available')
```

**Solution:** Ensure camera is publishing:
```bash
# Check topics
ros2 topic list | grep camera

# Echo camera data
ros2 topic echo /camera/image_raw --once
```

### Pipeline Not Ready

```bash
response: std_srvs.srv.Trigger_Response(success=False, message='Pipeline not ready')
```

**Solution:** Check pipeline initialization in node logs or restart the node.

---

## Performance Notes

- **Detection**: ~0.5-2s depending on image size and GPU availability
- **Classification**: ~0.1-0.5s per object
- **Grasp Generation**: ~0.2-1s depending on number of objects
- **Scene Graph**: ~0.1s for typical scenes

Use individual services for better performance when you don't need the full pipeline.

---

## Additional Resources

- Main README: `README.md`
- Quick Reference: `QUICK_REFERENCE.md`
- Pipeline Summary: `SAM_PIPELINE_SUMMARY.md`
- Installation Guide: `docs/INSTALL.md`
