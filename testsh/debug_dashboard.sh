#!/bin/bash
# Debug Dashboard - Check if dashboard is receiving data

echo "=========================================="
echo "Dashboard Debugging Tool"
echo "=========================================="
echo ""

echo "1. Checking if dashboard node is running..."
if ros2 node list | grep -q "benchmark_dashboard"; then
    echo "✓ Dashboard node is running"
else
    echo "✗ Dashboard node is NOT running"
    echo "  Start it with: ros2 run vision benchmark_dashboard"
    exit 1
fi

echo ""
echo "2. Checking dashboard topics..."
echo "Topics that should exist:"
echo "  - /vision/sam_detections (SAMDetections)"
echo "  - /vision/scene_understanding (SceneUnderstanding)"
echo "  - /benchmark/results (String)"
echo "  - /benchmark/data (String - dashboard publishes this)"
echo ""

ros2 topic list | grep -E "(sam_detections|scene_understanding|benchmark)"

echo ""
echo "3. Checking if /vision/sam_detections has any publishers..."
ros2 topic info /vision/sam_detections

echo ""
echo "4. Checking if dashboard is subscribed to /vision/sam_detections..."
ros2 topic info /vision/sam_detections | grep -A5 "Subscription count"

echo ""
echo "5. Testing if messages are being published..."
echo "Listening to /vision/sam_detections for 3 seconds..."
timeout 3 ros2 topic echo /vision/sam_detections --once 2>/dev/null && echo "✓ Messages detected!" || echo "✗ No messages detected in 3 seconds"

echo ""
echo "6. Checking dashboard's published data..."
echo "Listening to /benchmark/data for 2 seconds..."
timeout 2 ros2 topic echo /benchmark/data --once 2>/dev/null && echo "✓ Dashboard is publishing data!" || echo "✗ Dashboard not publishing data"

echo ""
echo "7. Checking HTTP server..."
if curl -s http://localhost:8080 > /dev/null; then
    echo "✓ HTTP server is accessible at http://localhost:8080"
else
    echo "✗ HTTP server is NOT accessible"
fi

echo ""
echo "8. Checking /api/data endpoint..."
echo "Sample of dashboard data:"
curl -s http://localhost:8080/api/data | head -c 200
echo ""
echo ""

echo "=========================================="
echo "Diagnostic complete!"
echo ""
echo "To see data in dashboard:"
echo "1. Make sure dashboard is running: ros2 run vision benchmark_dashboard"
echo "2. Launch your robot: ros2 launch ur_yt_sim final_project.launch.py mode:=real"
echo "3. IMPORTANT: Trigger object detection by calling the service:"
echo "   ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects"
echo "4. Open browser: http://localhost:8080"
echo "=========================================="
