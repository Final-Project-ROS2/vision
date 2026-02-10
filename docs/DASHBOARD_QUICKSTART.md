# 🎯 Benchmark Dashboard - Quick Reference

## 🚀 Quick Start (3 Steps)

```bash
# Step 1: Build the package
cd ~/final_project_ws
colcon build --packages-select vision --symlink-install
source install/setup.bash

# Step 2: Start the dashboard
./src/vision/testsh/start_benchmark_dashboard.sh

# Step 3: Open browser
# Dashboard opens automatically at http://localhost:8080
```

## 📊 Dashboard Sections

| Section | Data Displayed | Key Metrics |
|---------|----------------|-------------|
| **Pixel to Real** | Input (u,v) → Output (x,y,z) | Conversion count |
| **SAM Detection** | Bounding boxes, IoU, confidence | Avg IoU, Stability Rate, Avg Confidence |
| **CLIP Classification** | Labels, confidence scores | Top-1 Accuracy, Avg Confidence |
| **GraspNet** | Grasp poses, quality, width | Avg Quality Score, Avg Width |
| **Scene Understanding** | Objects, relations, descriptions | Spatial Accuracy, Adjacency Accuracy |

## 🎮 Common Commands

```bash
# Start dashboard
ros2 run vision benchmark_dashboard

# Generate test data
./src/vision/testsh/test_benchmark_dashboard.sh

# Run pipeline (generates data for all services)
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger

# Clear all data
ros2 service call /benchmark/clear_data std_srvs/srv/Trigger

# View raw data stream
ros2 topic echo /benchmark/data
```

## 🔧 Full Pipeline Setup

```bash
# Terminal 1: Pixel to Real Service
ros2 run vision pixel_to_real_service

# Terminal 2: SAM Detector
ros2 run vision simple_sam_detector

# Terminal 3: CLIP Classifier
ros2 run vision clip_classifier

# Terminal 4: GraspNet Detector
ros2 run vision graspnet_detector

# Terminal 5: Scene Understanding
ros2 run vision scene_understanding

# Terminal 6: Benchmark Dashboard
ros2 run vision benchmark_dashboard

# Terminal 7: Trigger pipeline
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
```

## 📈 Benchmark Metrics Explained

### SAM Metrics
- **IoU Score**: Intersection over Union with previous frame detection (0-1)
- **AP (IoU≥0.5)**: Pass if IoU ≥ 0.5 (COCO standard)
- **Stability Rate**: Percentage of detections with IoU ≥ 0.5

### CLIP Metrics
- **Top-1 Accuracy**: Correct label in top prediction (True/False)
- **Confidence**: Model's certainty (0-1)

### GraspNet Metrics
- **Quality Score**: Grasp success probability (0-1)
- **Grasp Width**: Gripper opening (meters)
- **Approach Direction**: top/side/angle

### Scene Understanding Metrics
- **Spatial Accuracy**: Quality of spatial relations (%)
- **Adjacency Accuracy**: Quality of proximity relations (%)

## 🎨 Color Coding

| Color | Meaning | Value Range |
|-------|---------|-------------|
| 🟢 Green | High confidence/quality | ≥ 0.7 |
| 🟠 Orange | Medium confidence/quality | 0.4 - 0.7 |
| 🔴 Red | Low confidence/quality | < 0.4 |

## 🔍 Troubleshooting

**No data showing?**
```bash
# Check if services are running
ros2 service list | grep vision

# Trigger pipeline to generate data
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
```

**Dashboard won't open?**
```bash
# Check if port 8080 is in use
netstat -tuln | grep 8080

# Check node is running
ros2 node list | grep benchmark
```

**Data not updating?**
- Check browser console (F12)
- Try refreshing the page
- Verify: `curl http://localhost:8080/api/data`

## 📁 Files Created

```
vision/
├── vision/
│   └── benchmark_dashboard.py          # ROS2 monitoring node
├── dashboard/
│   └── index.html                      # Web interface
├── testsh/
│   ├── start_benchmark_dashboard.sh    # Launcher script
│   └── test_benchmark_dashboard.sh     # Test data generator
└── docs/
    ├── BENCHMARK_DASHBOARD.md          # Full documentation
    └── DASHBOARD_QUICKSTART.md         # This file
```

## 💡 Tips

1. **Auto-refresh**: Dashboard updates every 2 seconds automatically
2. **Data limits**: Keeps last 1000 records per service (prevents memory issues)
3. **No code changes**: Original service nodes are unmodified
4. **Export data**: Subscribe to `/benchmark/data` topic for external analysis
5. **Multiple browsers**: Open dashboard in multiple tabs for different views

## 🎯 Use Cases

1. **Performance Testing**: Monitor detection accuracy and stability
2. **Algorithm Comparison**: Compare different model configurations
3. **System Debugging**: Identify pipeline bottlenecks
4. **Demo/Presentation**: Live visualization of system capabilities
5. **Data Collection**: Export benchmark data for analysis

---

**Need more help?** See full documentation: `docs/BENCHMARK_DASHBOARD.md`
