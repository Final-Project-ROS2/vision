#!/bin/bash
# Start Benchmark Dashboard
# Opens the dashboard automatically in your default browser

echo "======================================"
echo "Vision Benchmark Dashboard Launcher"
echo "======================================"
echo ""

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "ERROR: ROS2 not sourced. Please run:"
    echo "  source /opt/ros/<distro>/setup.bash"
    echo "  source ~/final_project_ws/install/setup.bash"
    exit 1
fi

# Activate vision_venv
echo "Activating vision_venv..."
if [ -d "/home/group11/vision_venv" ]; then
    source /home/group11/vision_venv/bin/activate
    echo "✓ vision_venv activated"
else
    echo "WARNING: vision_venv not found at /home/group11/vision_venv"
    echo "Dashboard will use system Python"
fi
echo ""

echo "Starting benchmark dashboard node..."
echo ""
echo "Dashboard will be available at:"
echo "  http://localhost:8080"
echo ""
echo "To clear benchmark data, run in another terminal:"
echo "  ros2 service call /benchmark/clear_data std_srvs/srv/Trigger"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "======================================"
echo ""

# Open browser after a short delay (in background)
(sleep 3 && xdg-open http://localhost:8080 2>/dev/null || open http://localhost:8080 2>/dev/null) &

# Start the dashboard node
ros2 run vision benchmark_dashboard
