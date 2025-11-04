#!/bin/bash

echo "======================================================================="
echo "Webcam Image Capture and Vision Pipeline Test"
echo "======================================================================="
echo ""
echo "This script demonstrates the new webcam image saving feature:"
echo "1. Camera service captures webcam frames"
echo "2. Images are automatically saved to src-webcam/"
echo "3. Vision pipeline processes the live camera feed"
echo "4. Service calls can detect objects from the current frame"
echo ""
echo "======================================================================="

# Navigate to workspace
cd /home/group11/final_project_ws
source install/local_setup.bash

echo ""
echo "📸 Starting camera service with image saving..."
echo "   - Images will be saved to: src/vision/build/vision/src-webcam/"
echo "   - Save interval: Every 30 frames (~1 second)"
echo ""

# Start camera service with image saving
ros2 run vision camera_service --ros-args \
  -p save_images:=true \
  -p save_interval:=30 &

CAMERA_PID=$!
sleep 5

echo ""
echo "🤖 Starting vision pipeline..."
echo ""

# Start vision pipeline
ros2 run vision sam_vision_pipeline &
VISION_PID=$!
sleep 3

echo ""
echo "======================================================================="
echo "✅ Both services are now running!"
echo "======================================================================="
echo ""
echo "Camera Service PID: $CAMERA_PID"
echo "Vision Pipeline PID: $VISION_PID"
echo ""
echo "To test object detection on current webcam frame:"
echo "  ros2 service call /vision/detect_objects std_srvs/srv/Trigger"
echo ""
echo "To view saved webcam images:"
echo "  ls -lh /home/group11/final_project_ws/src/vision/build/vision/src-webcam/"
echo ""
echo "To stop the services:"
echo "  pkill -f camera_service && pkill -f sam_vision_pipeline"
echo ""
echo "Press Ctrl+C to stop this script"
echo "======================================================================="

# Wait for user interrupt
wait
