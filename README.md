# Vision - ROS2 RGB Camera Viewer

ROS2 package that subscribes to `/camera/image_raw` and displays RGB images using OpenCV via CvBridge conversion.

## Quick Setup (WSL)
```bash
# 1. Setup workspace
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
cp -r /mnt/c/Users/Admin/Downloads/vision .

# 2. Install dependencies & build
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select vision
source install/setup.bash
```

## Usage

**Terminal 1 - Run the node:**
```bash
ros2 run vision show_rgb_image
```

**Terminal 2 - Control display:**
```bash
# Show single image
ros2 service call /show_rgb_image std_srvs/srv/Trigger

# Toggle continuous display on/off
ros2 service call /toggle_continuous_display std_srvs/srv/Trigger
```

## How it works
- **Subscribes** to `/camera/image_raw` (sensor_msgs/Image)
- **Converts** ROS images to OpenCV using `bridge.imgmsg_to_cv2(msg, 'bgr8')`
- **Displays** with OpenCV `cv2.imshow()`
- **Two modes**: single image on-demand or continuous feed



**Set up Gazebo**
```bash
# Terminal 1
colbon build
```


```bash
# Terminal 2
source venv/bin/activate
source install/setup.bash
ros2 launch ur_yt_sim spawn_ur5_camera_gripper_moveit.launch.py 

```

/final_project_ws
colcon build --packages-select vision --symlink-install

/final_project_ws/src/vision
source install/setup.bash && ros2 run vision clip_classifier



**Note:** Requires X11 forwarding in WSL for display: `export DISPLAY=:0`
