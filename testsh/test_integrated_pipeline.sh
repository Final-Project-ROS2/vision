#!/bin/bash
# Test Integrated Vision Pipeline: SAM Detection + CLIP Classification
# 
# This script demonstrates the complete pipeline:
# 1. Camera publishes to /camera/image_raw
# 2. SAM detector finds objects
# 3. CLIP classifier labels each detected object

echo "=========================================="
echo "Integrated Vision Pipeline Test"
echo "SAM Detection + CLIP Classification"
echo "=========================================="
echo ""

# Check if nodes are running
echo "Checking required nodes..."
echo ""

# Check for SAM detector
if ros2 service list | grep -q "/vision/detect_objects"; then
    echo "✅ SAM Detector is running (/vision/detect_objects available)"
else
    echo "❌ SAM Detector NOT running!"
    echo "   Start it with: ros2 run vision simple_sam_detector"
    echo ""
    read -p "Start SAM detector now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting SAM detector in background..."
        gnome-terminal -- bash -c "source install/setup.bash && ros2 run vision simple_sam_detector; exec bash" &
        sleep 3
    else
        echo "Cannot proceed without SAM detector. Exiting."
        exit 1
    fi
fi

# Check for CLIP classifier
if ros2 service list | grep -q "/vision/classify_detect"; then
    echo "✅ CLIP Classifier is running (/vision/classify_detect available)"
else
    echo "❌ CLIP Classifier NOT running!"
    echo "   Start it with: ros2 run vision clip_classifier"
    echo ""
    read -p "Start CLIP classifier now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting CLIP classifier in background..."
        gnome-terminal -- bash -c "source install/setup.bash && ros2 run vision clip_classifier; exec bash" &
        sleep 5
    else
        echo "Cannot proceed without CLIP classifier. Exiting."
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Running Integrated Pipeline Test"
echo "=========================================="
echo ""
echo "Pipeline: /camera/image_raw → SAM Detection → CLIP Classification"
echo ""
echo "Calling service: /vision/classify_detect"
echo ""
echo "This will:"
echo "  1. Get current camera image"
echo "  2. Detect objects with SAM"
echo "  3. Classify each detected region with CLIP"
echo "  4. Return JSON with all detections and classifications"
echo ""
echo "Press Enter to continue..."
read

# Call the integrated service
ros2 service call /vision/classify_detect std_srvs/srv/Trigger

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Check the OpenCV windows to see:"
echo "  - SAM Detector: Green boxes around detected objects"
echo "  - CLIP Classifier: Yellow boxes with classification labels"
echo ""
