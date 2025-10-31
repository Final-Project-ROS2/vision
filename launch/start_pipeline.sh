# Alternative bash script launch for systems without ROS2 launch
#!/bin/bash

# SAM Vision Pipeline Launch Script
# Alternative to ROS2 launch files for development/testing
# Pipeline: SAM (Meta) → CLIP → GraspNet → Scene Understanding

echo "🚀 Starting SAM Vision Pipeline"

# Set default parameters
USE_GAZEBO=${USE_GAZEBO:-true}
AUTO_PROCESS=${AUTO_PROCESS:-false}
PROCESSING_RATE=${PROCESSING_RATE:-1.0}
SAVE_RESULTS=${SAVE_RESULTS:-true}
DEBUG_VISUALIZATION=${DEBUG_VISUALIZATION:-true}

echo "Configuration:"
echo "  USE_GAZEBO: $USE_GAZEBO"
echo "  AUTO_PROCESS: $AUTO_PROCESS"
echo "  PROCESSING_RATE: $PROCESSING_RATE"
echo "  SAVE_RESULTS: $SAVE_RESULTS"
echo "  DEBUG_VISUALIZATION: $DEBUG_VISUALIZATION"

# Start Gazebo if requested
if [ "$USE_GAZEBO" = "true" ]; then
    echo "🌍 Starting Gazebo simulation..."
    ros2 launch gazebo_ros gazebo.launch.py world:=empty.world &
    GAZEBO_PID=$!
    sleep 5
fi

# Start camera simulation or hardware interface
echo "📷 Starting camera interface..."
# This could be replaced with actual camera launch
# ros2 run usb_cam usb_cam_node_exe &

# Start the SAM vision pipeline
echo "🔍 Starting SAM Vision Pipeline..."
ros2 run vision sam_vision_pipeline \
    --ros-args \
    -p auto_process:=$AUTO_PROCESS \
    -p processing_rate:=$PROCESSING_RATE \
    -p save_results:=$SAVE_RESULTS \
    -p debug_visualization:=$DEBUG_VISUALIZATION &
PIPELINE_PID=$!

# Start visualization
if [ "$DEBUG_VISUALIZATION" = "true" ]; then
    echo "📊 Starting visualization..."
    ros2 run image_view image_view --ros-args --remap image:=/vision/debug_image &
    IMAGE_VIEWER_PID=$!
    
    # Start RViz
    ros2 run rviz2 rviz2 -d $(ros2 pkg prefix vision)/share/vision/config/sam_pipeline.rviz &
    RVIZ_PID=$!
fi

echo "✅ SAM Vision Pipeline started!"
echo "Press Ctrl+C to stop all processes"

# Wait for interrupt
trap 'echo "🛑 Stopping all processes..."; kill $GAZEBO_PID $PIPELINE_PID $IMAGE_VIEWER_PID $RVIZ_PID 2>/dev/null; exit' INT
wait