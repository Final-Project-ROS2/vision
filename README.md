# Vision - ROS2 AI Vision Pipeline

TLDR: Complete ROS2 vision system integrating **SAM** (Segment Anything), **CLIP** (classification), **GraspNet** (grasp detection), and **Scene Understanding** for robotic perception.


## 🔄 Pipeline Flow

```
Camera Input → SAM Detection → CLIP Classification → GraspNet → Scene Understanding → Unified Output
```

1. **SAM** detects and segments objects
2. **CLIP** classifies detected regions
3. **GraspNet** generates 6D grasp poses
4. **Scene Understanding** analyzes spatial relationships
5. **Unified Pipeline** coordinates all modules and outputs JSON

---

height = 73.66cm or 29 inch

##  Quick Start

### 1. Build the Package

```bash
cd ~/final_project_ws
colcon build
source install/setup.bash
```

### 2. Activate Vision Environment

```bash
# From final_project_ws directory
source vision_venv/bin/activate
```

### 3. Launch Complete Pipeline

**Option A - Single Launch Command (Recommended):**
```bash
ros2 launch ur_yt_sim final_project.launch.py
```
argument

mode:=sim for simulation
mode:=real for depth cam

**Option B - Manual Node Startup:**
```bash
# Terminal 1: SAM Detector
ros2 run vision simple_sam_detector

# Terminal 2: CLIP Classifier
ros2 run vision clip_classifier

# Terminal 3: GraspNet Detector
ros2 run vision graspnet_detector

# Terminal 4: Scene Understanding
ros2 run vision scene_understanding

# Terminal 5: Pixel-to-Real Converter
ros2 run vision pixel_to_real_world_service

# Terminal 6: Unified Pipeline Orchestrator
ros2 run vision unified_pipeline
```

### 4. Trigger Vision Pipeline

```bash
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
```

## 📋 Setup Requirements

### Build Custom Interfaces
```bash
cd ~/final_project_ws
colcon build
source install/setup.bash
```

### Verify Custom Interfaces
```bash
ros2 interface show custom_interfaces/msg/SAMDetection
ros2 interface show custom_interfaces/msg/SAMDetections
ros2 interface show custom_interfaces/srv/PixelToReal
ros2 interface show custom_interfaces/srv/FindObject
```

### Install Python Dependencies
```bash
cd ~/final_project_ws/src/vision
pip install -r requirements.txt
```

### Setup Gazebo Simulation
```bash
# Terminal 1: Build
cd ~/final_project_ws
colcon build
source install/setup.bash

# Terminal 2: Launch Gazebo with UR5 robot
source vision_venv/bin/activate
source install/setup.bash
ros2 launch ur_yt_sim spawn_ur5_camera_gripper_moveit.launch.py
```

**Note:** Requires X11 forwarding in WSL for display: `export DISPLAY=:0`



---

# 🧠 Vision AI ROS2 Pipeline Architecture

This project integrates **SAM**, **CLIP**, **GraspNet**, and **Scene Understanding** into a complete ROS2-based vision perception pipeline.
Each module communicates via **ROS2 services** and **topics**.

---

## 📦 System Overview

| Node                          | Role                                                         | Key Services/Topics                                                                 |
| ----------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| **simple_sam_detector**       | Detects objects using SAM model                              | `/vision/run_pipeline`, `/vision/detect_objects`, `/vision/sam_detections`          |
| **clip_classifier**           | Classifies detected regions with CLIP                        | `/vision/classify_all`, `/vision/classify_bb`, `/vision/find_object`                |
| **graspnet_detector**         | Generates 6D grasp poses                                     | `/vision/detect_grasp`, `/vision/detect_grasp_bb`                                   |
| **scene_understanding**       | Analyzes spatial relationships                               | `/vision/understand_scene`, `/vision/scene_understanding`                           |
| **pixel_to_real_world_service** | Converts pixel coordinates to 3D world coordinates         | `/pixel_to_real_world`                                                              |
| **unified_pipeline**          | Orchestrates complete vision pipeline                        | `/vision/run_pipeline`                                                      |
| **find_object_service**       | High-level object search interface                           | `/vision/find_object_service`                                                       |
| **find_object_grasp_service** | Combined object search + grasp generation                    | `/vision/find_object_grasp_service`                                                 |

---

## ⚙️ Available ROS2 Services

## Core Pipeline Services

#### 1. **Unified Pipeline** (`/vision/run_pipeline`)
**Node:** `unified_pipeline`  
**Type:** `std_srvs/srv/Trigger`  
**Description:** Executes complete vision pipeline (SAM → CLIP → GraspNet → Scene Understanding) and saves results to JSON

```bash
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
```

---

## Object Detection Services (SAM)

#### 2. **SAM Object Detection** (`/vision/detect_objects`)
**Node:** `simple_sam_detector`  
**Type:** `std_srvs/srv/Trigger`  
**Description:** Detects all objects in current frame using SAM, returns bounding boxes and confidences

```bash
ros2 service call /vision/detect_objects std_srvs/srv/Trigger
```

#### 3. **Run SAM Pipeline** (`/vision/run_pipeline`)
**Node:** `simple_sam_detector`  
**Type:** `std_srvs/srv/Trigger`  
**Description:** Starts continuous SAM detection and publishes to `/vision/sam_detections` topic

```bash
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
```

#### 4. **Show Depth Image** (`/vision/show_depth_image`)
**Node:** `simple_sam_detector`  
**Type:** `std_srvs/srv/Trigger`  
**Description:** Displays depth camera visualization for debugging

```bash
ros2 service call /vision/show_depth_image std_srvs/srv/Trigger
```

---

## Classification Services (CLIP)

#### 5. **Classify All** (`/vision/classify_all`)
**Node:** `clip_classifier`  
**Type:** `std_srvs/srv/Trigger`  
**Description:** Classifies entire camera frame using CLIP model

```bash
ros2 service call /vision/classify_all std_srvs/srv/Trigger
```

#### 6. **Classify Bounding Box** (`/vision/classify_bb`)
**Node:** `clip_classifier`  
**Type:** `custom_interfaces/srv/ClassifyBBox`  
**Description:** Classifies specific region defined by bounding box

```bash
ros2 service call /vision/classify_bb custom_interfaces/srv/ClassifyBBox "{x1: 100, y1: 100, x2: 200, y2: 300}"
```

#### 7. **Find Object** (`/vision/find_object`)
**Node:** `clip_classifier`  
**Type:** `custom_interfaces/srv/FindObject`  
**Description:** Searches for specific object by name in detected regions

```bash
ros2 service call /vision/find_object custom_interfaces/srv/FindObject "{object_name: 'red_cube'}"
```

#### 8. **Find Object Service** (`/vision/find_object_service`)
**Node:** `find_object_service`  
**Type:** `custom_interfaces/srv/FindObject`  
**Description:** High-level object search with automatic detection + classification

```bash
ros2 service call /vision/find_object_service custom_interfaces/srv/FindObject "{object_name: 'drill'}"
```

---

## Grasp Generation Services (GraspNet)

#### 9. **Detect Grasp** (`/vision/detect_grasp`)
**Node:** `graspnet_detector`  
**Type:** `std_srvs/srv/Trigger`  
**Description:** Generates 6D grasp poses for all detected objects

```bash
ros2 service call /vision/detect_grasp std_srvs/srv/Trigger
```

#### 10. **Detect Grasp Bounding Box** (`/vision/detect_grasp_bb`)
**Node:** `graspnet_detector`  
**Type:** `custom_interfaces/srv/DetectGraspBBox`  
**Description:** Generates grasp pose for specific region

```bash
ros2 service call /vision/detect_grasp_bb custom_interfaces/srv/DetectGraspBBox "{x1: 100, y1: 100, x2: 200, y2: 300}"
```

#### 11. **Find Object Grasp** (`/vision/find_object_grasp_service`)
**Node:** `find_object_grasp_service`  
**Type:** `custom_interfaces/srv/FindObjectGrasp`  
**Description:** Combined service: find object + generate grasp pose

```bash
ros2 service call /vision/find_object_grasp_service custom_interfaces/srv/FindObjectGrasp "{object_name: 'wrench'}"
```

---

## Scene Understanding Services

#### 12. **Understand Scene** (`/vision/understand_scene`)
**Node:** `scene_understanding`  
**Type:** `std_srvs/srv/Trigger`  
**Description:** Analyzes spatial relationships between detected objects

```bash
ros2 service call /vision/understand_scene std_srvs/srv/Trigger
```

---

## Coordinate Transformation Services

#### 13. **Pixel to Real World** (`/pixel_to_real_world`)
**Node:** `pixel_to_real_world_service`  
**Type:** `custom_interfaces/srv/PixelToReal`  
**Description:** Converts 2D pixel coordinates to 3D world coordinates (x,y,z) based on the UR Arm and depth camera position

```bash
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"
```

---

## 📡 ROS2 Topics

### Published Topics

| Topic                         | Type                                      | Description                            |
| ----------------------------- | ----------------------------------------- | -------------------------------------- |
| `/vision/sam_detections`      | `custom_interfaces/msg/SAMDetections`     | Continuous SAM detection results       |
| `/vision/scene_understanding` | `custom_interfaces/msg/SceneUnderstanding` | Scene graph with spatial relations     |
| `/camera/image_raw`           | `sensor_msgs/Image`                       | RGB camera feed                        |
| `/camera/depth/image_raw`     | `sensor_msgs/Image`                       | Depth camera feed                      |
| `/camera/camera_info`         | `sensor_msgs/CameraInfo`                  | Camera calibration parameters          |

### Subscribed Topics

| Topic                     | Nodes Subscribing                                                     |
| ------------------------- | --------------------------------------------------------------------- |
| `/camera/image_raw`       | All vision nodes                                                      |
| `/camera/depth/image_raw` | `simple_sam_detector`, `graspnet_detector`, `pixel_to_real_world_service` |
| `/vision/sam_detections`  | `clip_classifier`, `graspnet_detector`, `scene_understanding`         |

---

## ⚙️ Node Details

| Node | Description | Command |
|------|-------------|---------|
| `simple_sam_detector` | Object detection using SAM | `ros2 run vision simple_sam_detector` |
| `clip_classifier` | Image classification using CLIP | `ros2 run vision clip_classifier` |
| `graspnet_detector` | 6D grasp pose generation | `ros2 run vision graspnet_detector` |
| `scene_understanding` | Spatial relationship analysis | `ros2 run vision scene_understanding` |
| `pixel_to_real_world_service` | Pixel to 3D coordinate conversion | `ros2 run vision pixel_to_real_world_service` |
| `unified_pipeline` | Complete pipeline orchestration | `ros2 run vision unified_pipeline` |
| `find_object_service` | High-level object search | `ros2 run vision find_object_service` |
| `find_object_grasp_service` | Object search + grasp generation | `ros2 run vision find_object_grasp_service` |

---

## 📊 Usage Examples

### Example 1: Find and Grasp an Object

```bash
# Start all nodes with launch file
ros2 launch vision unified_pipeline.launch.py

# Find object and get grasp pose in one call
ros2 service call /vision/find_object_grasp_service custom_interfaces/srv/FindObjectGrasp "{object_name: 'red_cube'}"
```

### Example 2: Complete Scene Analysis

```bash
# Run full pipeline
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger

# Results saved to: /home/group11/final_project_ws/src/vision/unified_pipeline_output.json
```

### Example 3: Custom Object Detection Workflow

```bash
# Step 1: Detect all objects
ros2 service call /vision/detect_objects std_srvs/srv/Trigger

# Step 2: Classify specific region
ros2 service call /vision/classify_bb custom_interfaces/srv/ClassifyBBox "{x1: 100, y1: 150, x2: 250, y2: 300}"

# Step 3: Generate grasp for that region
ros2 service call /vision/detect_grasp_bb custom_interfaces/srv/DetectGraspBBox "{x1: 100, y1: 150, x2: 250, y2: 300}"
```

---

# 📊 Benchmarking with Gazebo World

This section explains how to **benchmark the vision pipeline** using a series of 10 Gazebo simulation worlds.

There are 10 benchmark worlds located at:

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

```bash
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

This will launch both the Gazebo simulation and all vision nodes.

Running the launch command without the `world_file` arg will launch the default world.

---

## 📚 Additional Documentation

For more detailed information, see the `docs/` directory:

- **[QUICK_START.md](docs/QUICK_START.md)** - Getting started guide
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Complete API documentation
- **[UNIFIED_PIPELINE_SUMMARY.md](docs/UNIFIED_PIPELINE_SUMMARY.md)** - Unified pipeline details
- **[PIXEL_TO_REAL_WORLD_QUICK_REF.md](docs/PIXEL_TO_REAL_WORLD_QUICK_REF.md)** - Coordinate transformation guide
- **[BENCHMARK_DASHBOARD.md](docs/BENCHMARK_DASHBOARD.md)** - Benchmarking dashboard usage

---

## 🐛 Troubleshooting

### Issue: Services not responding
**Solution:** Ensure all nodes are running and sourced correctly
```bash
ros2 node list  # Check running nodes
ros2 service list  # Check available services
```

### Issue: Camera topics not publishing
**Solution:** Check Gazebo is running and camera plugin is loaded
```bash
ros2 topic list  # Should see /camera/image_raw and /camera/depth/image_raw
ros2 topic hz /camera/image_raw  # Check publishing rate
```

### Issue: CLIP model not loading
**Solution:** Ensure vision_venv is activated and in the file directory final_project_ws
```bash
source ~/final_project_ws/vision_venv/bin/activate
```

---

## 📄 License

Apache-2.0

---


