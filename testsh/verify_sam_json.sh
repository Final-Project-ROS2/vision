#!/bin/bash
# Quick verification script for SAM JSON output

echo "============================================"
echo "SAM Detector - JSON Output Verification"
echo "============================================"
echo ""

# Check if SAM detector is running
if pgrep -f simple_sam_detector > /dev/null; then
    echo "✅ SAM detector is running"
else
    echo "❌ SAM detector is NOT running"
    echo "   Start with: ros2 run vision simple_sam_detector"
    exit 1
fi

echo ""
echo "Calling /vision/detect_objects service..."
echo ""

# Call service and capture output
ros2 service call /vision/detect_objects std_srvs/srv/Trigger > /tmp/sam_output.txt 2>&1

# Check if successful
if grep -q "success: true" /tmp/sam_output.txt; then
    echo "✅ Service call successful"
    echo ""
    
    # Extract and show JSON
    echo "📋 JSON Response:"
    echo "----------------------------------------"
    grep -A 50 "message:" /tmp/sam_output.txt | sed 's/message: //' | head -n 40
    echo "----------------------------------------"
    echo ""
    
    # Count bounding boxes
    bbox_count=$(grep -o "bbox:" /tmp/sam_output.txt | wc -l)
    echo "📦 Found $bbox_count bounding boxes in JSON output"
    echo ""
    
    if [ $bbox_count -gt 0 ]; then
        echo "✅ VERIFIED: Bounding boxes are present in JSON output"
    else
        echo "⚠️  WARNING: No bounding boxes found"
    fi
else
    echo "❌ Service call failed"
    echo ""
    echo "Output:"
    cat /tmp/sam_output.txt
fi

echo ""
echo "============================================"
