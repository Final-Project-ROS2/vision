# How to Verify simple_sam_detector Publisher

## Current Implementation Status

The `simple_sam_detector.py` node has been configured to publish to `/vision/sam_detections` topic. Here's how to verify it's working:

---

## Method 1: Check Active Topics

### Step 1: Start the node
```bash
# Terminal 1 - Run the detector node
cd /home/group11/final_project_ws
source install/setup.bash
ros2 run vision simple_sam_detector
```

### Step 2: List all active topics
```bash
# Terminal 2 - Check topics
ros2 topic list
```

**Expected output:**
```
/camera/image_raw
/camera/depth/image_raw
/vision/sam_detections    ← Should appear here
```

---

## Method 2: Check Topic Info

### Get detailed information about the publisher
```bash
ros2 topic info /vision/sam_detections
```

**Expected output:**
```
Type: vision/msg/SAMDetections (or Image placeholder before build)
Publisher count: 1
Subscription count: 0
```

### Check the message type
```bash
ros2 topic type /vision/sam_detections
```

---

## Method 3: Echo Topic Data

### Listen to published messages
```bash
# Terminal 2 - Listen to the topic
ros2 topic echo /vision/sam_detections
```

### Trigger detection
```bash
# Terminal 3 - Call detection service
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

**Expected:** You should see message data appear in Terminal 2

---

## Method 4: Monitor Publish Rate

### Check how often messages are published
```bash
ros2 topic hz /vision/sam_detections
```

**Expected:** Shows rate when detections are triggered

---

## Method 5: Verify in Node Logs

When you call the detection service, check the node's terminal for:

```
[INFO] [...]: Publishing 3 detections to /vision/sam_detections
```

This confirms the `_publish_detections_ros()` method is being called.

---

## Method 6: Create a Test Subscriber

Create a simple subscriber to verify message reception:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # Placeholder type

class TestSubscriber(Node):
    def __init__(self):
        super().__init__('test_subscriber')
        self.subscription = self.create_subscription(
            Image,  # Will be SAMDetections after build
            '/vision/sam_detections',
            self.callback,
            10
        )
        self.get_logger().info('Waiting for messages on /vision/sam_detections...')
    
    def callback(self, msg):
        self.get_logger().info(f'Received detection message!')
        # After build with proper types:
        # self.get_logger().info(f'Received {len(msg.detections)} detections')

def main():
    rclpy.init()
    node = TestSubscriber()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

**Save as:** `test_subscriber.py` and run:
```bash
python3 test_subscriber.py
```

---

## Method 7: Use ROS2 Bag to Record

### Record the topic data
```bash
ros2 bag record /vision/sam_detections
```

### Trigger some detections
```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

### Check the bag file
```bash
ros2 bag info rosbag2_*
```

Should show recordings on `/vision/sam_detections`

---

## Method 8: Check RQT Graph

### Visualize the node connections
```bash
rqt_graph
```

**Expected:** You should see:
- `simple_sam_detector` node
- Arrow from node to `/vision/sam_detections` topic
- Shows publisher relationship

---

## Complete Test Workflow

### 1. Start Gazebo simulation
```bash
ros2 launch vision sam_gazebo_complete.launch.py
```

### 2. Start detector node (in new terminal)
```bash
cd /home/group11/final_project_ws
source install/setup.bash
ros2 run vision simple_sam_detector
```

### 3. Monitor the topic (in new terminal)
```bash
ros2 topic echo /vision/sam_detections
```

### 4. Trigger detection (in new terminal)
```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

### 5. Verify data appears in step 3's terminal

---

## Expected Behavior

### Before Detection:
- Topic exists but no data flowing
- `ros2 topic hz` shows "no messages received"

### After Detection Service Call:
- Message published to topic
- `ros2 topic echo` shows message data
- Node logs show "Publishing X detections to /vision/sam_detections"
- Subscriber receives the message

---

## Troubleshooting

### Issue: Topic doesn't appear
**Solution:** 
- Check node is running: `ros2 node list`
- Check for errors in node terminal
- Verify publisher was created in __init__

### Issue: No data on topic
**Solution:**
- Trigger detection: `ros2 service call /vision/detect_objects std_srvs/srv/Trigger`
- Check `_publish_detections_ros()` is called
- Look for errors in node logs

### Issue: Wrong message type
**Solution:**
- After building with CMakeLists.txt, uncomment the actual message code
- Replace `Image` placeholder with `SAMDetections`
- Rebuild package

---

## After Building Custom Messages

Once you build the package with the new messages, update `simple_sam_detector.py`:

```python
# Uncomment these lines in the file:
from vision.msg import SAMDetections, SAMDetection

# And uncomment the actual publishing code in _publish_detections_ros()
```

Then the topic will publish proper `SAMDetections` messages instead of placeholder.

---

## Quick Verification Commands

```bash
# Check if publisher exists
ros2 topic list | grep sam_detections

# Check publisher details
ros2 topic info /vision/sam_detections -v

# Listen for messages
ros2 topic echo /vision/sam_detections

# Trigger detection
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```
