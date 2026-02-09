# Debug CLIP find_object Service

## Quick Debug Commands

### 1. Check if CLIP is installed
```bash
python3 -c "import clip; print('CLIP installed:', clip.available_models())"
```

### 2. Check if nodes are running
```bash
# Should show: clip_classifier, simple_sam_detector, find_object_service_node
ros2 node list
```

### 3. Check if services are available
```bash
# Should show all these services:
ros2 service list | grep -E "(find_object|detect_objects|classify)"
```

### 4. Test find_object with verbose logging
```bash
# Terminal 1: Run CLIP classifier with debug logs
ros2 run vision clip_classifier --ros-args --log-level debug

# Terminal 2: Run SAM detector
ros2 run vision simple_sam_detector

# Terminal 3: Run find_object service
ros2 run vision find_object_service_node

# Terminal 4: Call the service
ros2 service call /find_object custom_interfaces/srv/FindObjectReal "{label: 'bowl'}"
```

## Common Issues & Solutions

### Issue 1: CLIP not installed or import error
**Symptoms:**
- `ModuleNotFoundError: No module named 'clip'`
- Node crashes on startup

**Solution:**
```bash
pip install --upgrade setuptools wheel
pip install git+https://github.com/openai/CLIP.git
pip install torch torchvision ftfy regex pillow
```

### Issue 2: Service not found
**Symptoms:**
- `Service /find_object not found`
- `waiting for service to become available`

**Solution:**
```bash
# Check which nodes are running
ros2 node list

# Start missing nodes:
ros2 run vision clip_classifier &
ros2 run vision simple_sam_detector &
ros2 run vision find_object_service_node &
```

### Issue 3: No detections / empty response
**Symptoms:**
- `success: False`
- `message: "No objects detected"`
- CLIP runs but finds nothing

**Debug:**
```bash
# 1. Check camera feed
ros2 topic echo /camera/image_raw --once

# 2. Test SAM detection directly
ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects

# 3. Check SAM output file
cat /tmp/sam_detections.json

# 4. Verify label is in candidate list
ros2 run vision clip_classifier --ros-args -p candidate_labels:="['bowl', 'beer_can', 'coke_can']"
```

### Issue 4: Label not found in detections
**Symptoms:**
- SAM detects objects but label doesn't match
- `Object 'X' not found in detections`

**Solution:**
```bash
# 1. Check what labels CLIP is using
# Edit vision/clip_classifier.py lines 117-165 to see candidate_labels

# 2. Test what CLIP sees
ros2 service call /vision/classify_bbox_filtered std_srvs/srv/Trigger

# 3. Try broader labels
ros2 service call /find_object custom_interfaces/srv/FindObjectReal "{label: 'bowl'}"
ros2 service call /find_object custom_interfaces/srv/FindObjectReal "{label: 'cup'}"
```

### Issue 5: CLIP model loading error
**Symptoms:**
- `RuntimeError: Model ... not found`
- Slow startup or timeout

**Debug:**
```bash
# Test CLIP model loading
python3 << 'EOF'
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Available models: {clip.available_models()}")

try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("✓ CLIP model loaded successfully")
except Exception as e:
    print(f"✗ Error loading CLIP: {e}")
EOF
```

### Issue 6: Service call timeout
**Symptoms:**
- Service hangs indefinitely
- No response after calling `/find_object`

**Debug:**
```bash
# 1. Check node logs
ros2 node info /clip_classifier
ros2 node info /find_object_service_node

# 2. Monitor service calls with verbose output
ros2 service call /find_object custom_interfaces/srv/FindObjectReal "{label: 'bowl'}" --verbose

# 3. Check if pipeline is stuck
ps aux | grep -E "(clip|sam|find_object)"

# 4. Restart all nodes
killall clip_classifier simple_sam_detector find_object_service_node
# Then restart them
```

## Step-by-Step Debug Workflow

### Step 1: Verify Installation
```bash
# Check Python packages
pip list | grep -E "(clip|torch|pillow)"

# Expected:
# clip              1.0
# torch             2.x.x
# pillow            10.x.x
```

### Step 2: Test CLIP in Isolation
```bash
python3 << 'EOF'
import clip
import torch
from PIL import Image
import numpy as np

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Create test image
test_image = Image.new('RGB', (640, 480), color='blue')
image_tensor = preprocess(test_image).unsqueeze(0).to(device)

# Test classification
text = clip.tokenize(["bowl", "cup", "bottle"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image_tensor, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
print("✓ CLIP working! Probabilities:", probs)
EOF
```

### Step 3: Test Services Individually
```bash
# A. Test SAM detection
ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects

# B. Test CLIP classification (after SAM detection)
ros2 service call /vision/classify_bbox_filtered std_srvs/srv/Trigger

# C. Test pixel_to_real conversion
ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{x: 320, y: 240}"

# D. Finally test integrated find_object
ros2 service call /find_object custom_interfaces/srv/FindObjectReal "{label: 'bowl'}"
```

### Step 4: Monitor with Debug Logs
```bash
# Terminal 1: CLIP with debug logging
ROS_LOG_LEVEL=DEBUG ros2 run vision clip_classifier

# Terminal 2: Watch for classification results
ros2 topic echo /vision/clip_classifications

# Terminal 3: Call service
ros2 service call /find_object custom_interfaces/srv/FindObjectReal "{label: 'bowl'}"
```

## Useful Debug Topics

### Monitor CLIP output
```bash
# Listen to classification results
ros2 topic echo /vision/clip_classifications

# Monitor camera feed
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

### Check detection data
```bash
# View SAM detections
ros2 topic echo /vision/sam_detections

# Check detection JSON file
cat /tmp/sam_detections.json | python3 -m json.tool
```

## Performance Checks

### Check if GPU is being used
```bash
# In Python console:
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Monitor processing time
```bash
# Add timing to clip_classifier.py find_object_callback
# Check logs for: "Classification took X.XX seconds"
```

## Quick Fix Script

Save as `debug_clip.sh`:
```bash
#!/bin/bash

echo "=== CLIP Debug Script ==="

echo -e "\n1. Checking CLIP installation..."
python3 -c "import clip; print('✓ CLIP installed')" 2>&1 || echo "✗ CLIP not installed"

echo -e "\n2. Checking nodes..."
ros2 node list | grep -E "(clip|sam|find_object)" || echo "✗ Nodes not running"

echo -e "\n3. Checking services..."
ros2 service list | grep -E "(find_object|detect|classify)" || echo "✗ Services not available"

echo -e "\n4. Checking camera topic..."
timeout 2 ros2 topic echo /camera/image_raw --once > /dev/null 2>&1 && echo "✓ Camera publishing" || echo "✗ No camera data"

echo -e "\n5. Testing CLIP model..."
python3 << 'EOF'
try:
    import clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✓ CLIP model loaded on {device}")
except Exception as e:
    print(f"✗ Error: {e}")
EOF

echo -e "\n=== Debug Complete ==="
```

Make executable and run:
```bash
chmod +x debug_clip.sh
./debug_clip.sh
```
