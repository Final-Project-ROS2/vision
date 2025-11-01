# ROS2 SAM Vision Pipeline - Quick Reference

## 🚀 Installation & Setup

```bash
# 1. Run automated installer
chmod +x install_pipeline.sh
./install_pipeline.sh

# 2. Manual setup (if needed)
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --packages-select vision
source install/setup.bash
```

## ⚡ Quick Start Commands

```bash
# Launch complete pipeline with Gazebo
ros2 launch vision dino_pipeline.launch.py use_gazebo:=true

# Launch for real camera
ros2 launch vision dino_pipeline.launch.py use_gazebo:=false

# Run demo
ros2 run vision vision_demo

# Manual node control
ros2 run vision dino_vision_pipeline
```

## 📡 ROS2 Topics & Services

### Key Topics
```bash
# Input topics (subscribe)
/camera/image_raw          # RGB camera feed
/camera/depth/image_raw    # Depth camera feed

# Output topics (publish)  
/vision/debug_image        # Annotated visualization
/vision/grasp_poses        # 6D grasp poses
```

### Services
```bash
# Process current scene
ros2 service call /vision/process_scene std_srvs/srv/Trigger

# Reset pipeline
ros2 service call /vision/reset_pipeline std_srvs/srv/Trigger
```

## 🔧 Configuration

### Launch Parameters
```bash
ros2 launch vision dino_pipeline.launch.py \
    use_gazebo:=true \
    auto_process:=false \
    processing_rate:=1.0 \
    save_results:=true \
    debug_visualization:=true
```

### Node Parameters
```bash
ros2 run vision dino_vision_pipeline --ros-args \
    -p auto_process:=true \
    -p processing_rate:=2.0 \
    -p save_results:=true
```

## 🐛 Debugging

### Check Topics
```bash
ros2 topic list
ros2 topic echo /camera/image_raw
ros2 topic hz /vision/debug_image
```

### Monitor Node
```bash
ros2 node info /dino_vision_pipeline
ros2 param list /dino_vision_pipeline
```

### View Images
```bash
ros2 run image_view image_view --ros-args --remap image:=/vision/debug_image
```

### RViz Visualization
```bash
ros2 run rviz2 rviz2 -d ~/ros2_ws/src/vision/config/dino_pipeline.rviz
```

## 📊 Pipeline Stages

1. **Detection** → Objects with bounding boxes
2. **Semantic** → Attributes and affordances  
3. **Grasp** → 6D poses with quality scores
4. **Scene** → Spatial relationships and understanding

## 🎯 Common Use Cases

### Process Single Image
```bash
# Trigger processing
ros2 service call /vision/process_scene std_srvs/srv/Trigger

# Check results
ls ~/ros2_vision_outputs/scene_*/
```

### Continuous Processing
```bash
# Enable auto-processing
ros2 param set /dino_vision_pipeline auto_process true
ros2 param set /dino_vision_pipeline processing_rate 1.0
```

### Save Results
```bash
# Results automatically saved to:
~/ros2_vision_outputs/scene_TIMESTAMP/
  ├── debug_visualization.jpg
  ├── pipeline_results.json
  ├── input_rgb.jpg
  └── input_depth.png
```

## 🔍 Troubleshooting

### No Camera Data
```bash
# Check camera topics
ros2 topic list | grep camera
ros2 topic echo --once /camera/image_raw
```

### Pipeline Not Working
```bash
# Check service availability
ros2 service list | grep vision
ros2 service type /vision/process_scene

# Check node status
ros2 node list | grep vision
```

### Performance Issues
```bash
# CPU mode (if GPU issues)
export CUDA_VISIBLE_DEVICES=""

# Reduce processing rate  
ros2 param set /dino_vision_pipeline processing_rate 0.5
```

### Missing Dependencies
```bash
# Reinstall Python packages
pip install -r Final-proj/config/requirements.txt

# Rebuild package
cd ~/ros2_ws && colcon build --packages-select vision
```

## 📝 File Locations

```
~/ros2_ws/src/vision/           # Package source
~/ros2_vision_outputs/          # Results output
~/.ros/log/                     # ROS logs
~/ros2_ws/install/vision/       # Built package
```

## 🎮 Gazebo Integration

### Launch with Custom World
```bash
ros2 launch vision dino_pipeline.launch.py world_file:=workshop.world
```

### Spawn Objects
```bash
# Use Gazebo GUI or launch files to add objects for testing
```

## 📈 Performance Tuning

### GPU Optimization
```bash
# Check GPU usage
nvidia-smi

# Monitor memory
watch -n 1 nvidia-smi
```

### CPU Optimization  
```bash
# Set number of threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 🆘 Getting Help

```bash
# Package info
ros2 pkg xml vision

# Node help
ros2 run vision dino_vision_pipeline --help

# Launch file help  
ros2 launch vision dino_pipeline.launch.py --show-args
```

---
**Ready to run advanced robotic vision! 🤖👁️**