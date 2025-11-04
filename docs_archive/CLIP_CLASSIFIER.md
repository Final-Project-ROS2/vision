# CLIP Image Classifier - ROS2 Node

## 📋 Overview

A dedicated CLIP-based image classification node that:
- ✅ Subscribes to `/camera/image_raw` (sensor_msgs/Image)
- ✅ Uses OpenAI's `clip-vit-base-patch32` model
- ✅ Displays live classification with OpenCV window
- ✅ Exports results in structured JSON schema with embeddings
- ✅ Supports custom candidate labels

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers pillow
```

### 2. Run the Classifier

**Default labels** (robot, tool, part, container, etc.):
```bash
cd /home/group11/final_project_ws
source install/setup.bash
ros2 run vision clip_classifier
```

**Custom labels**:
```bash
ros2 run vision clip_classifier --labels "cat,dog,car,airplane,person"
```

### 3. Call Classification Service

```bash
ros2 service call /vision/classify_image std_srvs/srv/Trigger
```

---

## 📊 JSON Schema Output

### Response Format

```json
{
  "pipeline": "single_clip",
  "model": "openai/clip-vit-base-patch32",
  "input": {
    "image_path": "frame_000042",
    "candidate_labels": ["cat", "dog", "car", "airplane"]
  },
  "output": {
    "top_prediction": {
      "label": "dog",
      "confidence": 0.91
    },
    "all_predictions": [
      { "label": "dog", "confidence": 0.91 },
      { "label": "cat", "confidence": 0.05 },
      { "label": "car", "confidence": 0.03 },
      { "label": "airplane", "confidence": 0.01 }
    ],
    "embedding": {
      "image_vector": [0.123, -0.512, 0.341, ...],
      "text_vectors": [
        {"label": "cat", "vector": [0.130, -0.498, 0.333, ...]},
        {"label": "dog", "vector": [0.125, -0.510, 0.339, ...]}
      ],
      "similarity_method": "cosine"
    },
    "metadata": {
      "timestamp": "2025-11-03T14:15:00Z",
      "processing_time_ms": 78,
      "device": "cuda:0"
    }
  }
}
```

### Schema Fields

| Field | Type | Description |
|-------|------|-------------|
| `pipeline` | string | Always "single_clip" |
| `model` | string | CLIP model name |
| `input.image_path` | string | Frame identifier |
| `input.candidate_labels` | array | Labels to classify against |
| `output.top_prediction` | object | Highest confidence prediction |
| `output.all_predictions` | array | All predictions sorted by confidence |
| `output.embedding.image_vector` | array | 512-dim image embedding |
| `output.embedding.text_vectors` | array | Text embeddings for each label |
| `output.embedding.similarity_method` | string | Always "cosine" |
| `metadata.timestamp` | string | ISO 8601 UTC timestamp |
| `metadata.processing_time_ms` | int | Classification time in milliseconds |
| `metadata.device` | string | "cuda:0" or "cpu" |

---

## 🔧 CLIP Processing Flow

```
Gazebo Camera
     ↓
/camera/image_raw (ROS Image)
     ↓
bridge.imgmsg_to_cv2(msg, 'bgr8')
     ↓
Convert BGR → RGB
     ↓
CLIP Preprocessing:
  • Resize to 224x224
  • Normalize (ImageNet stats)
  • Convert to tensor
     ↓
CLIP Model:
  • Image encoder (Vision Transformer)
  • Text encoder (Transformer)
  • Compute similarity scores
     ↓
Outputs:
  • Softmax probabilities
  • Image embeddings (512-dim)
  • Text embeddings (512-dim per label)
     ↓
Visualization:
  • cv2.imshow() with predictions
  • JSON schema export
```

---

## 👁️ OpenCV Window Display

```
┌─────────────────────────────────────────────┐
│ CLIP Classifier | Frame: 000042             │
│                                             │
│          [Camera Image]                     │
│                                             │
│                                   1. dog: 91%│
│                                   2. cat: 5% │
│                                   3. car: 3% │
│─────────────────────────────────────────────│
│ Top: dog                                    │
│ 91.0%                                       │
└─────────────────────────────────────────────┘
```

---

## 📝 Usage Examples

### Example 1: Basic Classification

```bash
# Terminal 1: Start classifier
ros2 run vision clip_classifier

# Terminal 2: Classify current frame
ros2 service call /vision/classify_image std_srvs/srv/Trigger
```

### Example 2: Custom Labels

```bash
# Classify as specific object types
ros2 run vision clip_classifier --labels "hammer,screwdriver,wrench,pliers"

# Call service
ros2 service call /vision/classify_image std_srvs/srv/Trigger
```

### Example 3: Extract Top Prediction

```bash
ros2 service call /vision/classify_image std_srvs/srv/Trigger | \
  grep -A 1000 'message:' | sed '1d' | \
  jq '.output.top_prediction'
```

### Example 4: Get Image Embeddings

```bash
ros2 service call /vision/classify_image std_srvs/srv/Trigger | \
  grep -A 1000 'message:' | sed '1d' | \
  jq '.output.embedding.image_vector' | head -20
```

### Example 5: Save Classification Results

```bash
ros2 service call /vision/classify_image std_srvs/srv/Trigger | \
  grep -A 1000 'message:' | sed '1d' > clip_result_$(date +%s).json
```

---

## 🐛 Troubleshooting

### Issue: "CLIP not available"

**Solution:**
```bash
pip install torch torchvision transformers pillow

# For CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Model loading slow

**Solution:**
- First run downloads ~600MB model from HuggingFace
- Subsequent runs load from cache (~/.cache/huggingface)
- Use `device: cuda` for faster inference (if GPU available)

### Issue: Out of memory

**Solution:**
```python
# Edit clip_classifier.py
self.device = "cpu"  # Force CPU instead of CUDA
```

### Issue: No image received

**Solution:**
```bash
# Check camera topic
ros2 topic hz /camera/image_raw

# View camera
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

---

## 🎓 Understanding CLIP Embeddings

### Image Vector
- **Dimension**: 512
- **Purpose**: Semantic representation of image content
- **Use**: Compare images, search, clustering

### Text Vectors
- **Dimension**: 512 per label
- **Purpose**: Semantic representation of each candidate label
- **Use**: Zero-shot classification, text-image matching

### Cosine Similarity
```python
similarity = dot(image_vector, text_vector) / (norm(image_vector) * norm(text_vector))
confidence = softmax(similarity)
```

---

## 🔗 Integration Example

### Python ROS2 Node

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import json

class VisionClient(Node):
    def __init__(self):
        super().__init__('vision_client')
        self.client = self.create_client(Trigger, '/vision/classify_image')
    
    def classify(self):
        request = Trigger.Request()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response.success:
            data = json.loads(response.message)
            top_pred = data['output']['top_prediction']
            print(f"Detected: {top_pred['label']} ({top_pred['confidence']:.0%})")
            return data
        return None

# Usage
node = VisionClient()
result = node.classify()
```

---

## 📊 Performance

| Metric | CPU | GPU (CUDA) |
|--------|-----|------------|
| **First inference** | ~2-3 sec | ~1 sec |
| **Subsequent** | ~200-300 ms | ~50-100 ms |
| **Model size** | 600 MB | 600 MB |
| **Memory usage** | ~1.5 GB | ~2 GB |
| **Embedding dim** | 512 | 512 |

---

## 🎯 Use Cases

### 1. Object Recognition
```bash
ros2 run vision clip_classifier --labels "box,cylinder,sphere,cone"
```

### 2. Tool Classification
```bash
ros2 run vision clip_classifier --labels "hammer,screwdriver,wrench,pliers,drill"
```

### 3. Scene Understanding
```bash
ros2 run vision clip_classifier --labels "kitchen,office,warehouse,factory,outdoor"
```

### 4. Quality Control
```bash
ros2 run vision clip_classifier --labels "defective,good,damaged,incomplete"
```

---

## 📦 Files

- **Main Node**: `/src/vision/vision/clip_classifier.py`
- **Setup Entry**: Added to `setup.py` as `clip_classifier`
- **Documentation**: `/src/vision/CLIP_CLASSIFIER.md`

---

## 🔄 Comparison: CLIP vs SAM Detector

| Feature | SAM Detector | CLIP Classifier |
|---------|-------------|-----------------|
| **Purpose** | Object detection/segmentation | Image classification |
| **Output** | Bounding boxes, masks | Class labels, confidence |
| **Speed** | ~30 Hz | ~3-10 Hz |
| **Model** | OpenCV contours | CLIP ViT-B/32 |
| **Embeddings** | ❌ | ✅ 512-dim vectors |
| **Custom labels** | ❌ | ✅ Zero-shot |
| **GPU benefit** | Low | High |

---

## 💡 Advanced Tips

### 1. Combine with SAM Detector
```python
# First detect objects with SAM
sam_detections = call_service('/vision/detect_objects')

# Then classify each detected region
for bbox in sam_detections:
    # Crop region
    # Classify with CLIP
    # Combine results
```

### 2. Use Embeddings for Similarity
```python
# Get embedding for reference image
ref_embedding = get_image_embedding(ref_image)

# Compare with new images
similarity = cosine_similarity(ref_embedding, new_embedding)
```

### 3. Custom Text Prompts
```bash
# Use descriptive prompts instead of single words
ros2 run vision clip_classifier --labels \
  "a photo of a red tool,a photo of a blue part,a photo of a green container"
```

---

## 📚 CLIP Model Details

- **Architecture**: Vision Transformer (ViT-B/32)
- **Image size**: 224×224
- **Patch size**: 32×32
- **Layers**: 12 transformer blocks
- **Hidden dim**: 768
- **Embedding dim**: 512
- **Training data**: 400M image-text pairs
- **Zero-shot**: Can classify any object described in text

---

**Happy Classifying! 🎯**
