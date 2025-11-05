#!/bin/bash
# Test script for integrated SAM+CLIP pipeline

echo "=================================="
echo "Testing Integrated SAM+CLIP Pipeline"
echo "=================================="
echo ""

# Check if camera is publishing
echo "1. Checking camera topic..."
if ros2 topic list | grep -q "/camera/image_raw"; then
    echo "   ✅ Camera topic /camera/image_raw found"
else
    echo "   ❌ Camera topic /camera/image_raw NOT found"
    echo "   Please start the camera first!"
    exit 1
fi

# Check if service exists
echo ""
echo "2. Checking service..."
if ros2 service list | grep -q "/vision/process_pipeline"; then
    echo "   ✅ Service /vision/process_pipeline found"
else
    echo "   ⚠️  Service /vision/process_pipeline NOT found"
    echo "   The node needs to be (re)started"
    echo ""
    echo "   Start with: ros2 run vision sam_clip_pipeline"
    exit 1
fi

# Call the service
echo ""
echo "3. Calling service /vision/process_pipeline..."
echo "   This will:"
echo "   - Detect objects with SAM"
echo "   - Show SAM results (press any key to continue)"
echo "   - Classify with CLIP"
echo "   - Show final results"
echo "   - Print JSON to terminal"
echo ""
echo "   Calling service now..."
ros2 service call /vision/process_pipeline std_srvs/srv/Trigger

echo ""
echo "=================================="
echo "Test complete!"
echo "=================================="
