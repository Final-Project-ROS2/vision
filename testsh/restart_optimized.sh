#!/bin/bash
# Quick restart script for optimized SAM+CLIP pipeline

echo "============================================"
echo "SAM+CLIP Pipeline - Quick Restart"
echo "============================================"

# Kill any existing instances
echo "🛑 Stopping existing instances..."
pkill -f simple_sam_detector
pkill -f sam_clip_pipeline
pkill -f clip_classifier
sleep 1

# Rebuild with symlink
echo "🔨 Rebuilding with symlink-install..."
cd /home/group11/final_project_ws
colcon build --packages-select vision --symlink-install 2>&1 | grep -E "(Starting|Finished|Summary)"

# Source
echo "📦 Sourcing workspace..."
source install/setup.bash

echo ""
echo "✅ Ready to run!"
echo ""
echo "Start nodes in separate terminals:"
echo "  Terminal 1: ros2 run vision simple_sam_detector"
echo "  Terminal 2: ros2 run vision sam_clip_pipeline"
echo "  Terminal 3: ros2 service call /vision/classify_detect std_srvs/srv/Trigger"
echo ""
echo "Expected response time: 2-5 seconds ⚡"
echo "============================================"
