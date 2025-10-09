# Vision ROS2 Package - WSL Quick Start

## Setup in WSL
```bash
# 1. Install ROS2 Humble (if not already installed)
sudo apt update && sudo apt install ros-humble-desktop

# 2. Create workspace and copy package
mkdir -p ~/ros2_ws/src
cp -r /mnt/c/Users/Admin/Downloads/vision ~/ros2_ws/src/

# 3. Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# 4. Build package
colcon build --packages-select vision

# 5. Source workspace
source install/setup.bash
```

## Run
```bash
# Start the image viewer node
ros2 run vision show_rgb_image

# In another terminal, call service to show image
ros2 service call /show_rgb_image std_srvs/srv/Trigger
```

**Note:** Make sure you have a camera publishing to `/camera/image_raw` topic.