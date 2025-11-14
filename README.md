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



## Set up Gazebo
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




## All Sub and Pub Node 
Run build 

```bash
cd ~/final_project_ws
colcon build --packages-select vision
source install/setup.bash
```


vision vnev setting in final_project_ws
```bash
source vision_venv/bin/activate
```
build custom interface

```bash
cd ~/final_project_ws
colcon build --packages-select custom_interfaces
source install/setup.bash
```
Verify the custom interface
```bash
ros2 interface show custom_interfaces/msg/SAMDetection
ros2 interface show custom_interfaces/msg/SAMDetections
```



---

# 🧠 Vision AI ROS2 Pipeline

This project integrates **SAM**, **CLIP**, **GraspNet**, and **Scene Understanding** into a complete ROS2-based vision perception pipeline.
Each module communicates via **ROS2 services** and **topics**.

All functions are **services** (called on demand), except `/vision/run_pipeline`, which acts as a **continuous pipeline trigger** (message-based, called once to start full pipeline flow).

---

## 📦 Overview

| Node                    | Role                                                         | Key Topics/Services                                                                                    |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| **simple_sam_detector** | Detects objects and publishes regions for downstream modules | `/vision/run_pipeline`, `/vision/detect_objects`, `/vision/show_depth_image`, `/vision/sam_detections` |
| **clip_classifier**     | Classifies detected regions (using CLIP model)               | `/vision/classify_all`, `/vision/classify_bb`, subscribes to `/vision/sam_detections`                  |
| **graspnet_detector**   | Estimates grasp poses for detected objects                   | `/vision/detect_grasp`, `/vision/detect_grasp_bb`, subscribes to `/vision/sam_detections`              |
| **scene_understanding** | Analyzes scene relationships and context                     | `/vision/understand_scene`, subscribes to `/vision/sam_detections`                                     |

---

## ⚙️ Node Details

### 1. **SAM Detector Node (`simple_sam_detector`)**

Detects objects using SAM (Segment Anything Model).

**Services**

| Name                       | Description                                                                       | Example                                                                        |
| -------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `/vision/run_pipeline`     | Trigger full SAM pipeline and continuously publish to `/vision/sam_detections`    | `ros2 service call /vision/run_pipeline std_srvs/srv/Trigger`                  |
| `/vision/detect_objects`   | Detect objects in one frame and return bounding boxes, confidences, and distances | `ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects` |
| `/vision/show_depth_image` | Display depth camera visualization                                                | `ros2 service call /vision/show_depth_image std_srvs/srv/Trigger`              |

**Setup**

```
Terminal 1: ros2 run vision simple_sam_detector
```

---

### 2. **CLIP Vision Classifier Node (`clip_classifier`)**

Classifies image regions using the CLIP model.

**Services**

| Name                   | Description                                                                          | Example                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `/vision/classify_all` | Classify the entire camera frame                                                     | `ros2 service call /vision/classify_all std_srvs/srv/Trigger`                                                     |
| `/vision/classify_bb`  | Classify a specific bounding box region                                              | `ros2 service call /vision/classify_bb custom_interfaces/srv/ClassifyBBox "{x1: 100, y1: 100, x2: 200, y2: 300}"` |
| `/vision/run_pipeline` | Subscribes to `/vision/sam_detections` and automatically classifies detected regions | `ros2 service call /vision/run_pipeline std_srvs/srv/Trigger`                                                     |

**Setup**

```
Terminal 1: ros2 run vision simple_sam_detector
Terminal 2: ros2 run vision clip_classifier
```

---

### 3. **GraspNet Detector Node (`graspnet_detector`)**

Estimates grasp poses for detected objects using GraspNet.

**Services**

| Name                      | Description                                                                   | Example                                                                                                                  |
| ------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `/vision/detect_grasp`    | Compute grasp poses for all detected objects                                  | `ros2 service call /vision/detect_grasp custom_interfaces/srv/DetectGrasps`                                              |
| `/vision/detect_grasp_bb` | Compute grasp pose within a specific bounding box                             | `ros2 service call /vision/detect_grasp_bb custom_interfaces/srv/DetectGraspBBox "{x1: 100, y1: 100, x2: 200, y2: 300}"` |
| `/vision/run_pipeline`    | Subscribes to `/vision/sam_detections` and automatically runs grasp detection | `ros2 service call /vision/run_pipeline std_srvs/srv/Trigger`                                                            |

**Setup**

```
Terminal 1: ros2 run vision simple_sam_detector
Terminal 2: ros2 run vision clip_classifier
Terminal 3: ros2 run vision graspnet_detector
```

---

### 4. **Scene Understanding Node (`scene_understanding`)**

Analyzes spatial relationships and context between detected objects.

**Services**

| Name                       | Description                                                               | Example                                                                            |
| -------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `/vision/understand_scene` | Get scene-level understanding (calls `/vision/detect_objects` internally) | `ros2 service call /vision/understand_scene custom_interfaces/srv/UnderstandScene` |
| `/vision/run_pipeline`     | Subscribes to `/vision/sam_detections` and runs full scene analysis       | `ros2 service call /vision/run_pipeline std_srvs/srv/Trigger`                      |

**Publishes**

* `/vision/scene_understanding` → `SceneUnderstanding` message

**Setup**

```
Terminal 1: ros2 run vision simple_sam_detector
Terminal 2: ros2 run vision clip_classifier
Terminal 3: ros2 run vision graspnet_detector
Terminal 4: ros2 run vision scene_understanding
```

---

## 🧩 Pipeline Flow Summary

```
SAM → CLIP → GraspNet → Scene Understanding
```

* `/vision/run_pipeline` triggers continuous message flow
* Each downstream node subscribes to `/vision/sam_detections`
* All other services are **on-demand** (per-frame or per-request basis)

---

Below is the **Benchmarking with Gazebo** section you can append directly to your README.
I preserved your structure and added clear, step-by-step instructions so anyone can reproduce benchmarking easily.

---

# 📊 Benchmarking with Gazebo

This section explains how to **benchmark the vision pipeline** using a series of 10 Gazebo simulation worlds.

there are 10 benchmark worlds located at:

```
~/final_project_ws/src/ur_yt_sim/worlds/test_world_x.world
```

Where `x` ∈ **1 to 10**, e.g.,

```
test_world_1.world
test_world_2.world
...
test_world_10.world
```

These worlds contain different object arrangements that allow for testing the vision node under various visual conditions.
Details of the objects in each world can be found in this [Google Sheet](https://docs.google.com/spreadsheets/d/1E-lBc7FS0EGegg0zXIx3ohNcED27fkgQaXMT66xFjTg/edit?usp=sharing)

---

### 🔧 1. Sourcing the ROS2 Workspace

Open a terminal, then go to the final_project_ws Workspace by running

```
cd ~/final_project_ws/
```

To source the workspace, run:

```bash
source install/setup.bash
```

### 🔧 2. Launching a Benchmark World

The main Gazebo simulation launch file accepts a launch argument:

```
world_file:=<name_of_world_file>
```

To benchmark a specific world, run:

```bash
ros2 launch ur_yt_sim spawn_ur5_camera_gripper_moveit.launch.py world_file:=test_world_1.world
```

Example: launch world 7

```bash
ros2 launch ur_yt_sim spawn_ur5_gripper_moveit.launch.py world_file:=test_world_7.world
```

This will launch both the Gazebo simulation and all vision nodes

Running the launch command without the `world_file` arg will launch the default world

---

### ✔️ Summary

You can benchmark your vision pipeline by:

1. Running each Gazebo world
2. Calling the appropriate vision services
3. Collecting time and performance metrics





