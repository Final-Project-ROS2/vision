#!/bin/bash
# Test script to generate sample benchmark data
# Triggers vision pipeline and individual services to populate the dashboard

# Activate vision_venv if available
if [ -d "/home/group11/vision_venv" ]; then
    source /home/group11/vision_venv/bin/activate
fi

echo "======================================"
echo "Benchmark Dashboard Test Script"
echo "======================================"
echo ""

# Check if services are available
echo "Checking if vision services are running..."
echo ""

services=(
    "/vision/run_pipeline"
    "/vision/detect_objects"
    "/pixel_to_real"
)

missing_services=false
for service in "${services[@]}"; do
    if ros2 service list | grep -q "$service"; then
        echo "✓ $service is available"
    else
        echo "✗ $service is NOT available"
        missing_services=true
    fi
done

echo ""

if [ "$missing_services" = true ]; then
    echo "WARNING: Some services are not running."
    echo "Please start all vision nodes first:"
    echo "  Terminal 1: ros2 run vision pixel_to_real_service"
    echo "  Terminal 2: ros2 run vision simple_sam_detector"
    echo "  Terminal 3: ros2 run vision clip_classifier"
    echo "  Terminal 4: ros2 run vision graspnet_detector"
    echo "  Terminal 5: ros2 run vision scene_understanding"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "======================================"
echo "Running test sequence..."
echo "======================================"
echo ""

# Test 1: Run full pipeline
echo "[1/5] Running full vision pipeline..."
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
sleep 2
echo "✓ Pipeline executed"
echo ""

# Test 2: Test pixel to real conversions
echo "[2/5] Testing pixel to real conversions..."
test_pixels=(
    "320 240"
    "305 95"
    "466 160"
    "150 200"
    "320 500"
)

for coords in "${test_pixels[@]}"; do
    u=$(echo $coords | cut -d' ' -f1)
    v=$(echo $coords | cut -d' ' -f2)
    echo "  Converting pixel ($u, $v)..."
    ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: $u, v: $v}" &
done
wait
echo "✓ Pixel conversions completed"
echo ""

# Test 3: Trigger SAM detection
echo "[3/5] Triggering SAM detection..."
ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects
sleep 1
echo "✓ SAM detection completed"
echo ""

# Test 4: Run pipeline again for more data
echo "[4/5] Running pipeline again..."
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
sleep 2
echo "✓ Second pipeline run completed"
echo ""

# Test 5: Test scene understanding
echo "[5/5] Testing scene understanding..."
ros2 service call /vision/understand_scene std_srvs/srv/Trigger
sleep 1
echo "✓ Scene understanding completed"
echo ""

echo "======================================"
echo "Test sequence completed!"
echo "======================================"
echo ""
echo "Check the dashboard at http://localhost:8080"
echo "You should see data in all sections now."
echo ""
echo "To generate more data, run this script again or use:"
echo "  ros2 service call /vision/run_pipeline std_srvs/srv/Trigger"
echo ""
