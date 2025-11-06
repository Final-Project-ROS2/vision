#!/bin/bash
# verify_sam_publisher.sh
# Quick script to verify simple_sam_detector publisher

echo "================================================================================"
echo "SAM Detector Publisher Verification Script"
echo "================================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Checking if simple_sam_detector node is running...${NC}"
if ros2 node list | grep -q "simple_sam_detector"; then
    echo -e "${GREEN}✓ Node is running${NC}"
else
    echo -e "${RED}✗ Node is NOT running${NC}"
    echo "  Start it with: ros2 run vision simple_sam_detector"
    exit 1
fi
echo ""

echo -e "${YELLOW}Step 2: Checking if /vision/sam_detections topic exists...${NC}"
if ros2 topic list | grep -q "/vision/sam_detections"; then
    echo -e "${GREEN}✓ Topic exists${NC}"
else
    echo -e "${RED}✗ Topic does NOT exist${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}Step 3: Getting topic information...${NC}"
ros2 topic info /vision/sam_detections
echo ""

echo -e "${YELLOW}Step 4: Checking topic type...${NC}"
TOPIC_TYPE=$(ros2 topic type /vision/sam_detections)
echo "  Type: $TOPIC_TYPE"
echo ""

echo -e "${YELLOW}Step 5: Checking publisher count...${NC}"
PUB_COUNT=$(ros2 topic info /vision/sam_detections | grep "Publisher count" | awk '{print $3}')
if [ "$PUB_COUNT" -ge 1 ]; then
    echo -e "${GREEN}✓ Found $PUB_COUNT publisher(s)${NC}"
else
    echo -e "${RED}✗ No publishers found${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}Step 6: Checking if detection service exists...${NC}"
if ros2 service list | grep -q "/vision/detect_objects"; then
    echo -e "${GREEN}✓ Detection service exists${NC}"
else
    echo -e "${RED}✗ Detection service NOT found${NC}"
    exit 1
fi
echo ""

echo "================================================================================"
echo -e "${GREEN}ALL CHECKS PASSED!${NC}"
echo "================================================================================"
echo ""
echo "To verify data publishing:"
echo "  1. In one terminal, run:"
echo "       ros2 topic echo /vision/sam_detections"
echo ""
echo "  2. In another terminal, trigger detection:"
echo "       ros2 service call /vision/detect_objects std_srvs/srv/Trigger"
echo ""
echo "  3. You should see message data in the first terminal"
echo ""
echo "================================================================================"
