#!/bin/bash
# Start Benchmark Dashboard with vision_venv
# Explicitly uses vision_venv Python interpreter

echo "======================================"
echo "Vision Benchmark Dashboard (venv)"
echo "======================================"
echo ""

VENV_PATH="/home/group11/vision_venv"
PYTHON_BIN="$VENV_PATH/bin/python3"
SCRIPT_PATH="/home/group11/final_project_ws/src/vision/vision/benchmark_dashboard.py"

# Check if vision_venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: vision_venv not found at $VENV_PATH"
    echo ""
    echo "Please create the virtual environment first:"
    echo "  python3 -m venv $VENV_PATH"
    echo "  source $VENV_PATH/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if Python exists in venv
if [ ! -f "$PYTHON_BIN" ]; then
    echo "ERROR: Python not found in vision_venv"
    echo "Expected: $PYTHON_BIN"
    exit 1
fi

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "ERROR: ROS2 not sourced. Please run:"
    echo "  source /opt/ros/<distro>/setup.bash"
    echo "  source ~/final_project_ws/install/setup.bash"
    exit 1
fi

# Activate venv
echo "Activating vision_venv..."
source $VENV_PATH/bin/activate
echo "✓ Using Python: $(which python3)"
echo "✓ Python version: $(python3 --version)"
echo ""

# Check required packages
echo "Checking required packages..."
python3 -c "import rclpy; print('✓ rclpy installed')" 2>/dev/null || echo "✗ rclpy not found"
python3 -c "import cv2; print('✓ opencv-python installed')" 2>/dev/null || echo "✗ opencv-python not found"
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

# Run the dashboard with venv Python
$PYTHON_BIN $SCRIPT_PATH
