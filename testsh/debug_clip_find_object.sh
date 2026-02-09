#!/bin/bash

echo "======================================"
echo "  CLIP find_object Debug Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if in virtual environment
echo ""
echo "0. Checking Virtual Environment..."
if [ -z "$VIRTUAL_ENV" ]; then
    print_status() { echo -e "${RED}✗${NC} $2"; }
    print_status 1 "NOT in virtual environment"
    echo -e "${YELLOW}  Fix: source vision_venv/bin/activate${NC}"
    echo -e "${YELLOW}  Then re-run this script${NC}"
    echo ""
    echo "Quick fix:"
    echo -e "  ${GREEN}cd ~/final_project_ws/src/vision${NC}"
    echo -e "  ${GREEN}source vision_venv/bin/activate${NC}"
    echo -e "  ${GREEN}./testsh/debug_clip_find_object.sh${NC}"
    exit 1
else
    echo -e "${GREEN}✓${NC} Virtual environment active: $VIRTUAL_ENV"
fi

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

echo ""
echo "1. Checking CLIP Installation..."
python3 -c "import clip; print('  Available models:', clip.available_models())" 2>&1
if [ $? -eq 0 ]; then
    print_status 0 "CLIP is installed"
else
    print_status 1 "CLIP is NOT installed"
    echo -e "${YELLOW}  Fix: pip install --upgrade setuptools wheel${NC}"
    echo -e "${YELLOW}       pip install git+https://github.com/openai/CLIP.git${NC}"
fi

echo ""
echo "2. Checking CLIP Model Loading..."
python3 << 'EOF'
try:
    import clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"  Device: {device}")
    print(f"  Model: ViT-B/32 loaded successfully")
except Exception as e:
    print(f"  Error: {e}")
    exit(1)
EOF
if [ $? -eq 0 ]; then
    print_status 0 "CLIP model loads successfully"
else
    print_status 1 "CLIP model failed to load"
fi

echo ""
echo "3. Checking ROS2 Nodes..."
ros2 node list > /tmp/nodes.txt 2>&1
if grep -q "clip_classifier" /tmp/nodes.txt; then
    print_status 0 "clip_classifier node is running"
else
    print_status 1 "clip_classifier node is NOT running"
    echo -e "${YELLOW}  Fix: ros2 run vision clip_classifier${NC}"
fi

if grep -q "simple_sam_detector" /tmp/nodes.txt; then
    print_status 0 "simple_sam_detector node is running"
else
    print_status 1 "simple_sam_detector node is NOT running"
    echo -e "${YELLOW}  Fix: ros2 run vision simple_sam_detector${NC}"
fi

if grep -q "find_object_service_node" /tmp/nodes.txt; then
    print_status 0 "find_object_service_node is running"
else
    print_status 1 "find_object_service_node is NOT running"
    echo -e "${YELLOW}  Fix: ros2 run vision find_object_service_node${NC}"
fi

echo ""
echo "4. Checking ROS2 Services..."
ros2 service list > /tmp/services.txt 2>&1
if grep -q "/find_object" /tmp/services.txt; then
    print_status 0 "/find_object service available"
else
    print_status 1 "/find_object service NOT available"
fi

if grep -q "/vision/detect_objects" /tmp/services.txt; then
    print_status 0 "/vision/detect_objects service available"
else
    print_status 1 "/vision/detect_objects service NOT available"
fi

if grep -q "/vision/find_object" /tmp/services.txt; then
    print_status 0 "/vision/find_object service available"
else
    print_status 1 "/vision/find_object service NOT available"
fi

echo ""
echo "5. Checking Camera Topic..."
timeout 2 ros2 topic echo /camera/image_raw --once > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status 0 "Camera is publishing images"
else
    print_status 1 "Camera is NOT publishing"
    echo -e "${YELLOW}  Check: Is Gazebo running?${NC}"
    echo -e "${YELLOW}  Check: ros2 topic list | grep camera${NC}"
fi

echo ""
echo "6. Checking SAM Detection Output..."
if [ -f /tmp/sam_detections.json ]; then
    detections=$(cat /tmp/sam_detections.json | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('detections', [])))" 2>/dev/null)
    if [ ! -z "$detections" ]; then
        print_status 0 "SAM detections file exists ($detections objects)"
    else
        print_status 1 "SAM detections file exists but empty"
    fi
else
    print_status 1 "SAM detections file NOT found"
    echo -e "${YELLOW}  Run: ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects${NC}"
fi

echo ""
echo "======================================"
echo "  Quick Test Commands"
echo "======================================"
echo ""
echo "Test detection pipeline:"
echo -e "  ${GREEN}ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects${NC}"
echo ""
echo "Test find_object (integrated):"
echo -e "  ${GREEN}ros2 service call /find_object custom_interfaces/srv/FindObjectReal \"{label: 'bowl'}\"${NC}"
echo ""
echo "Test find_object (CLIP only):"
echo -e "  ${GREEN}ros2 service call /vision/find_object custom_interfaces/srv/FindObject \"{label: 'bowl'}\"${NC}"
echo ""
echo "View current classifications:"
echo -e "  ${GREEN}ros2 service call /vision/classify_bbox_filtered std_srvs/srv/Trigger${NC}"
echo ""
echo "======================================"

# Cleanup
rm -f /tmp/nodes.txt /tmp/services.txt
