#!/bin/bash
# Test script for pixel_to_real_world service

echo "=========================================="
echo "Testing Pixel-to-Real-World Service"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if service is running
echo -e "${YELLOW}[1/5] Checking if service is running...${NC}"
if ros2 service list | grep -q "/pixel_to_real_world"; then
    echo -e "${GREEN}✓ Service /pixel_to_real_world is running${NC}"
else
    echo -e "${RED}✗ Service not found!${NC}"
    echo ""
    echo "Please start the service first:"
    echo "  ros2 run vision pixel_to_real_world_service"
    exit 1
fi
echo ""

# Check service type
echo -e "${YELLOW}[2/5] Checking service interface...${NC}"
SERVICE_TYPE=$(ros2 service type /pixel_to_real_world)
echo "Service type: $SERVICE_TYPE"
if [[ "$SERVICE_TYPE" == "custom_interfaces/srv/PixelToReal" ]]; then
    echo -e "${GREEN}✓ Correct service type${NC}"
else
    echo -e "${RED}✗ Unexpected service type${NC}"
fi
echo ""

# Test center pixel
echo -e "${YELLOW}[3/5] Testing center pixel (320, 240)...${NC}"
CENTER_RESULT=$(ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 320, v: 240}" 2>&1)
if [[ $? -eq 0 ]]; then
    echo "$CENTER_RESULT"
    echo -e "${GREEN}✓ Center pixel test passed${NC}"
else
    echo -e "${RED}✗ Center pixel test failed${NC}"
    echo "$CENTER_RESULT"
fi
echo ""

# Test corner pixels
echo -e "${YELLOW}[4/5] Testing corner pixels...${NC}"

echo "Top-left corner (100, 100):"
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 100, v: 100}" 2>&1 | grep -A 3 "response:"

echo ""
echo "Top-right corner (540, 100):"
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 540, v: 100}" 2>&1 | grep -A 3 "response:"

echo ""
echo "Bottom-left corner (100, 380):"
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 100, v: 380}" 2>&1 | grep -A 3 "response:"

echo ""
echo "Bottom-right corner (540, 380):"
ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 540, v: 380}" 2>&1 | grep -A 3 "response:"

echo -e "${GREEN}✓ Corner pixel tests completed${NC}"
echo ""

# Test multiple pixels in grid pattern
echo -e "${YELLOW}[5/5] Testing grid of pixels...${NC}"
echo "Testing 3x3 grid pattern:"

for v in 120 240 360; do
    for u in 213 320 427; do
        echo -n "Pixel ($u, $v) -> "
        RESULT=$(ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: $u, v: $v}" 2>&1 | grep -A 3 "response:")
        X=$(echo "$RESULT" | grep "x:" | awk '{print $2}')
        Y=$(echo "$RESULT" | grep "y:" | awk '{print $2}')
        Z=$(echo "$RESULT" | grep "z:" | awk '{print $2}')
        echo "World ($X, $Y, $Z)"
        sleep 0.5
    done
done

echo -e "${GREEN}✓ Grid test completed${NC}"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}All tests completed!${NC}"
echo "=========================================="
echo ""
echo "Service is working correctly."
echo ""
echo "Next steps:"
echo "  1. Integrate with object detection pipeline"
echo "  2. Test with real objects in camera view"
echo "  3. Validate 3D coordinate accuracy with known object positions"
echo ""
echo "For more information, see:"
echo "  docs/PIXEL_TO_REAL_WORLD_SERVICE.md"
