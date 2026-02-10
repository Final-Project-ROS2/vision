# Vision Benchmark Dashboard

Real-time monitoring and benchmarking dashboard for the ROS2 Vision Pipeline services.

## Features

The dashboard monitors and displays benchmark data for all vision services:

### 📍 Pixel to Real Conversion
- **Input**: Pixel coordinates (u, v)
- **Output**: World coordinates (x, y, z) in meters
- Tracks all conversion calls with timestamps

### 🎯 SAM Object Detection
- **Object ID**: Unique identifier for each detection
- **Bounding Box**: (x1, y1, x2, y2) coordinates
- **Center Point**: (u, v) pixel coordinates
- **Confidence**: Detection confidence score [0-1]
- **IoU Score**: Intersection over Union with previous frame
- **AP (IoU≥0.5)**: COCO-style Average Precision metric (Pass/Fail)
- **Distance**: Object distance in centimeters
- **Metrics**: Average IoU, Stability Rate, Average Confidence

### 🏷️ CLIP Classification
- **Label**: Predicted object class
- **Confidence**: Classification confidence [0-1]
- **Top-1 Accuracy**: Correctness indicator (True/False/N/A)
- **Bounding Box**: Region of interest
- **Metrics**: Top-1 Accuracy Rate, Average Confidence

### 🤖 Grasp Detection (GraspNet)
- **Object ID**: Associated detected object
- **Pixel Position**: (u, v) grasp point in image
- **World Position**: (x, y, z) grasp point in 3D space
- **Quality Score**: Grasp quality metric [0-1]
- **Grasp Width**: Gripper opening width in meters
- **Approach Angle**: Grasp orientation/direction
- **Metrics**: Average Quality Score, Average Grasp Width

### 🌐 Scene Understanding
- **Scene ID**: Unique scene identifier
- **Total Objects**: Number of detected objects
- **Relations**: Spatial relationships between objects
  - Subject → Relation → Object (with confidence)
- **Scene Description**: Natural language summary
- **Spatial Accuracy**: Accuracy of spatial relations (%)
- **Adjacency Accuracy**: Accuracy of near/touching relations (%)
- **Metrics**: Total Objects, Total Relations, Spatial/Adjacency Accuracy

## Installation

1. **Build the package:**
   ```bash
   cd ~/final_project_ws
   colcon build --packages-select vision --symlink-install
   source install/setup.bash
   ```

2. **No additional dependencies** - Uses built-in Python HTTP server

## Usage

### Quick Start

```bash
# Option 1: Use the convenience script (auto-opens browser)
./testsh/start_benchmark_dashboard.sh

# Option 2: Run directly
ros2 run vision benchmark_dashboard
```

The dashboard will be available at: **http://localhost:8080**

### Running with Pipeline Services

For full monitoring, run all vision services:

```bash
# Terminal 1: Start SAM detector
ros2 run vision simple_sam_detector

# Terminal 2: Start CLIP classifier
ros2 run vision clip_classifier

# Terminal 3: Start GraspNet detector
ros2 run vision graspnet_detector

# Terminal 4: Start Scene Understanding
ros2 run vision scene_understanding

# Terminal 5: Start Pixel-to-Real service
ros2 run vision pixel_to_real_service

# Terminal 6: Start Benchmark Dashboard
ros2 run vision benchmark_dashboard
```

Then trigger the pipeline:
```bash
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
```

### Using the Dashboard

1. **Auto-refresh**: Dashboard updates every 2 seconds automatically
2. **Clear Data**: Use the "Clear All Data" button or run:
   ```bash
   ros2 service call /benchmark/clear_data std_srvs/srv/Trigger
   ```
3. **Export Data**: Subscribe to the data topic:
   ```bash
   ros2 topic echo /benchmark/data
   ```

## Dashboard Features

### Real-time Statistics Cards
- Total service calls
- Count of each service type
- Color-coded confidence levels:
  - 🟢 Green: High confidence (≥0.7)
  - 🟠 Orange: Medium confidence (0.4-0.7)
  - 🔴 Red: Low confidence (<0.4)

### Interactive Tables
- Scrollable data tables (up to 50 most recent records)
- Hover effects for better readability
- Timestamp formatting
- Color-coded metrics

### Performance Metrics
- **SAM**: Average IoU, Stability Rate, Average Confidence
- **CLIP**: Top-1 Accuracy, Average Confidence
- **GraspNet**: Average Quality Score, Average Grasp Width
- **Scene**: Spatial Accuracy, Adjacency Accuracy

## Data Collection

The dashboard is **non-invasive** - it doesn't modify any existing service nodes. It works by:

1. **Subscribing to topics**: Monitors `/vision/sam_detections` and `/vision/scene_understanding`
2. **Publishing data**: Broadcasts collected data on `/benchmark/data` topic
3. **HTTP API**: Serves data via `/api/data` endpoint for the web interface

## API Endpoints

### GET /api/data
Returns complete benchmark data in JSON format:

```json
{
  "pixel_to_real": [...],
  "sam_detections": [...],
  "clip_classifications": [...],
  "grasp_detections": [...],
  "scene_understanding": [...],
  "metadata": {
    "start_time": "2026-02-10T...",
    "total_calls": 142
  }
}
```

### Service: /benchmark/clear_data
Clears all stored benchmark data:
```bash
ros2 service call /benchmark/clear_data std_srvs/srv/Trigger
```

## Extending the Dashboard

To add custom metrics or data collection:

1. **Modify `benchmark_dashboard.py`**: Add new data collection methods
2. **Update `dashboard/index.html`**: Add new visualization tables
3. **Rebuild**: `colcon build --packages-select vision --symlink-install`

## Troubleshooting

**Dashboard not loading?**
- Check if port 8080 is available: `netstat -tuln | grep 8080`
- Verify node is running: `ros2 node list | grep benchmark`

**No data showing?**
- Ensure vision services are running
- Trigger the pipeline: `ros2 service call /vision/run_pipeline std_srvs/srv/Trigger`
- Check topic connections: `ros2 topic info /vision/sam_detections`

**Data not updating?**
- Check browser console for JavaScript errors (F12)
- Verify `/api/data` endpoint: `curl http://localhost:8080/api/data`

## Performance Notes

- Data is limited to **1000 most recent records** per service type to prevent memory issues
- Tables show **50 most recent records** for optimal rendering
- Auto-refresh interval: 2 seconds (can be modified in HTML)
- HTTP server runs on a separate thread to avoid blocking ROS2 callbacks

## License

Apache-2.0

## Author

Final Project - Group 11
