#!/bin/bash
# start_sam_detector.sh - Start the simple_sam_detector node

echo "=========================================="
echo "Starting simple_sam_detector Node"
echo "=========================================="
echo ""

# Check if already running
if ros2 node list 2>/dev/null | grep -q "simple_sam_detector"; then
    echo "⚠️  simple_sam_detector is already running!"
    echo ""
    echo "If you want to restart it:"
    echo "  1. Find the process: ps aux | grep simple_sam_detector"
    echo "  2. Kill it: kill <PID>"
    echo "  3. Run this script again"
    exit 0
fi

# Source the workspace
cd /home/group11/final_project_ws
source install/setup.bash

echo "Starting simple_sam_detector node..."
echo ""
echo "The node will:"
echo "  - Subscribe to /camera/image_raw"
echo "  - Subscribe to /camera/depth/image_raw"
echo "  - Publish to /vision/sam_detections"
echo "  - Provide service /vision/detect_objects"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

ros2 run vision simple_sam_detector
