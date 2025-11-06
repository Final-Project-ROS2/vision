#!/bin/bash
# Quick test script for SAM detector publisher

echo "=========================================="
echo "Testing SAM Detector Publisher"
echo "=========================================="
echo ""

# Check if node is running
echo "1. Checking if simple_sam_detector is running..."
if ros2 node list 2>/dev/null | grep -q "simple_sam_detector"; then
    echo "   ✓ Node is running"
else
    echo "   ✗ Node is NOT running"
    echo ""
    echo "Start it with:"
    echo "   ros2 run vision simple_sam_detector"
    exit 1
fi
echo ""

# Check for topic
echo "2. Checking for /vision/sam_detections topic..."
if ros2 topic list 2>/dev/null | grep -q "/vision/sam_detections"; then
    echo "   ✓ Topic found!"
else
    echo "   ✗ Topic NOT found"
    echo ""
    echo "This might mean the publisher isn't working."
    exit 1
fi
echo ""

# Show topic info
echo "3. Topic information:"
ros2 topic info /vision/sam_detections
echo ""

# Show topic type
echo "4. Topic type:"
ros2 topic type /vision/sam_detections
echo ""

echo "=========================================="
echo "✓ Publisher is working correctly!"
echo "=========================================="
echo ""
echo "To see published data:"
echo "   ros2 topic echo /vision/sam_detections"
echo ""
echo "To trigger detection:"
echo "   ros2 service call /vision/detect_objects std_srvs/srv/Trigger"
