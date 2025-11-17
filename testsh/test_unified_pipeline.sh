#!/bin/bash
# Test script for Unified Vision Pipeline

echo "========================================"
echo "Unified Vision Pipeline Test"
echo "========================================"
echo ""

# Check if all required nodes are running
echo "Checking if all vision nodes are running..."
echo ""

REQUIRED_NODES=(
    "/simple_sam_detector"
    "/clip_classifier"
    "/graspnet_detector"
    "/scene_understanding"
    "/pixel_to_real_server"
    "/unified_pipeline"
)

MISSING_NODES=()

for node in "${REQUIRED_NODES[@]}"; do
    if ros2 node list | grep -q "$node"; then
        echo "✓ $node is running"
    else
        echo "✗ $node is NOT running"
        MISSING_NODES+=("$node")
    fi
done

echo ""

if [ ${#MISSING_NODES[@]} -ne 0 ]; then
    echo "ERROR: Missing nodes:"
    for node in "${MISSING_NODES[@]}"; do
        echo "  - $node"
    done
    echo ""
    echo "Please start all required nodes first:"
    echo "  ros2 launch vision unified_pipeline.launch.py"
    exit 1
fi

echo "All nodes are running!"
echo ""

# Check if all required services are available
echo "Checking if all vision services are available..."
echo ""

REQUIRED_SERVICES=(
    "/vision/detect_objects"
    "/vision/classify_bbox_filtered"
    "/vision/detect_grasp"
    "/vision/understand_scene"
    "/pixel_to_real"
    "/vision/run_unified_pipeline"
)

MISSING_SERVICES=()

for service in "${REQUIRED_SERVICES[@]}"; do
    if ros2 service list | grep -q "$service"; then
        echo "✓ $service is available"
    else
        echo "✗ $service is NOT available"
        MISSING_SERVICES+=("$service")
    fi
done

echo ""

if [ ${#MISSING_SERVICES[@]} -ne 0 ]; then
    echo "ERROR: Missing services:"
    for service in "${MISSING_SERVICES[@]}"; do
        echo "  - $service"
    done
    exit 1
fi

echo "All services are available!"
echo ""

# Run unified pipeline
echo "========================================"
echo "Running Unified Vision Pipeline..."
echo "========================================"
echo ""

ros2 service call /vision/run_unified_pipeline std_srvs/srv/Trigger

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Check output files in:"
echo "  ~/unified_pipeline_outputs/"
echo ""
echo "View latest result:"
echo "  cat ~/unified_pipeline_outputs/\$(ls -t ~/unified_pipeline_outputs/ | head -1)"
echo ""
