#!/bin/bash
# Pipeline Health Check Script
# Verifies that all components are working correctly

echo "=========================================="
echo "Vision Pipeline Health Check"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Camera topic
echo "1. Checking /camera/image_raw topic..."
if ros2 topic list | grep -q "/camera/image_raw"; then
    echo -e "${GREEN}✅ /camera/image_raw exists${NC}"
    
    # Check if publishing
    timeout 2s ros2 topic hz /camera/image_raw > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Camera is publishing${NC}"
    else
        echo -e "${YELLOW}⚠️  Camera topic exists but no data (timeout after 2s)${NC}"
        echo "   Start Gazebo with: ros2 launch vision sam_gazebo_complete.launch.py"
    fi
else
    echo -e "${RED}❌ /camera/image_raw NOT found${NC}"
    echo "   Start Gazebo with: ros2 launch vision sam_gazebo_complete.launch.py"
fi
echo ""

# Check 2: SAM Detector service
echo "2. Checking SAM Detector service..."
if ros2 service list | grep -q "/vision/detect_objects"; then
    echo -e "${GREEN}✅ /vision/detect_objects service available${NC}"
    
    # Try to call the service
    echo "   Testing SAM detector response..."
    timeout 5s ros2 service call /vision/detect_objects std_srvs/srv/Trigger > /tmp/sam_test.txt 2>&1
    if [ $? -eq 0 ]; then
        if grep -q "success: true" /tmp/sam_test.txt; then
            echo -e "${GREEN}✅ SAM detector responding successfully${NC}"
            # Count detections
            DETECTIONS=$(grep -o "\"detections\"" /tmp/sam_test.txt | wc -l)
            echo "   Found detection data in response"
        else
            echo -e "${YELLOW}⚠️  SAM detector returned success: false${NC}"
            grep "message:" /tmp/sam_test.txt | head -1
        fi
    else
        echo -e "${YELLOW}⚠️  SAM detector service timeout (5s)${NC}"
        echo "   Check if simple_sam_detector is processing images"
    fi
    rm -f /tmp/sam_test.txt
else
    echo -e "${RED}❌ /vision/detect_objects NOT available${NC}"
    echo "   Start with: ros2 run vision simple_sam_detector"
fi
echo ""

# Check 3: CLIP Classifier services
echo "3. Checking CLIP Classifier services..."
if ros2 service list | grep -q "/vision/classify_all"; then
    echo -e "${GREEN}✅ /vision/classify_all service available${NC}"
else
    echo -e "${RED}❌ /vision/classify_all NOT available${NC}"
    echo "   Start with: ros2 run vision clip_classifier"
fi

if ros2 service list | grep -q "/vision/classify_detect"; then
    echo -e "${GREEN}✅ /vision/classify_detect service available${NC}"
else
    echo -e "${RED}❌ /vision/classify_detect NOT available${NC}"
    echo "   Start with: ros2 run vision clip_classifier"
fi
echo ""

# Check 4: Node status
echo "4. Checking running nodes..."
NODES=$(ros2 node list 2>/dev/null)

if echo "$NODES" | grep -q "simple_sam_detector"; then
    echo -e "${GREEN}✅ simple_sam_detector node running${NC}"
else
    echo -e "${RED}❌ simple_sam_detector node NOT running${NC}"
fi

if echo "$NODES" | grep -q "clip_classifier"; then
    echo -e "${GREEN}✅ clip_classifier node running${NC}"
else
    echo -e "${RED}❌ clip_classifier node NOT running${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""

ERRORS=0

# Count issues
if ! ros2 topic list | grep -q "/camera/image_raw"; then
    ERRORS=$((ERRORS+1))
fi

if ! ros2 service list | grep -q "/vision/detect_objects"; then
    ERRORS=$((ERRORS+1))
fi

if ! ros2 service list | grep -q "/vision/classify_detect"; then
    ERRORS=$((ERRORS+1))
fi

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All components are ready!${NC}"
    echo ""
    echo "You can now test the integrated pipeline:"
    echo "  ros2 service call /vision/classify_detect std_srvs/srv/Trigger"
    echo ""
else
    echo -e "${YELLOW}⚠️  Found $ERRORS missing component(s)${NC}"
    echo ""
    echo "To fix, make sure all components are running:"
    echo "  1. Gazebo: ros2 launch vision sam_gazebo_complete.launch.py"
    echo "  2. SAM Detector: ros2 run vision simple_sam_detector"
    echo "  3. CLIP Classifier: ros2 run vision clip_classifier"
    echo ""
fi

# Detailed diagnostic if requested
if [ "$1" == "--verbose" ] || [ "$1" == "-v" ]; then
    echo ""
    echo "=========================================="
    echo "Detailed Diagnostics"
    echo "=========================================="
    echo ""
    
    echo "All ROS2 nodes:"
    ros2 node list
    echo ""
    
    echo "All ROS2 topics:"
    ros2 topic list
    echo ""
    
    echo "All ROS2 services:"
    ros2 service list | grep vision
    echo ""
fi
