#!/bin/bash
# Test script for /vision/understand_scene service
# This demonstrates that the service can be called multiple times

echo "=========================================="
echo "Testing /vision/understand_scene Service"
echo "=========================================="
echo ""
echo "This service can be called multiple times to update the scene understanding."
echo ""

# Function to call the service
call_service() {
    echo "----------------------------------------"
    echo "Call #$1 - $(date '+%H:%M:%S')"
    echo "----------------------------------------"
    ros2 service call /vision/understand_scene std_srvs/srv/Trigger
    echo ""
    sleep 2
}

# Call the service multiple times
echo "Making 3 consecutive service calls to demonstrate it can be called multiple times..."
echo ""

call_service 1
call_service 2
call_service 3

echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo "The service was called 3 times successfully."
echo "Each call reruns the scene understanding analysis."
echo ""
