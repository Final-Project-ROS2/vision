#!/bin/bash
# Setup and launch script for WSL

echo "=== Vision Package Setup & Launch ==="

# Set display for WSL
export DISPLAY=:0

# Check if in ROS2 workspace
if [ ! -f "install/setup.bash" ]; then
    echo "Error: Run this from your ROS2 workspace root (~/ros2_ws)"
    echo "Make sure you've built the package with: colcon build --packages-select vision"
    exit 1
fi

# Source the workspace
echo "Sourcing workspace..."
source install/setup.bash

# Verify package is available
if ! ros2 pkg list | grep -q "^vision$"; then
    echo "Error: vision package not found. Please build it first:"
    echo "  colcon build --packages-select vision"
    exit 1
fi

echo "✓ Package found and workspace sourced"
echo ""
echo "Starting RGB Image Viewer..."
echo "Available services:"
echo "  - /show_rgb_image (show single image)"
echo "  - /toggle_continuous_display (toggle live feed)"
echo ""
echo "Press Ctrl+C to stop the node"
echo ""

ros2 run vision show_rgb_image