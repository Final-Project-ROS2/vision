# 🤖 CLIP Classifier - Quick Reference

## ✅ What You Built

A dedicated CLIP image classification node:
- Uses OpenAI's CLIP model (clip-vit-base-patch32)
- Subscribes to `/camera/image_raw`
- Outputs structured JSON with predictions + embeddings
- Shows live classification in OpenCV window

---

## 🚀 Essential Commands

### 1. Run CLIP Classifier

**Default labels** (robot, tool, part, container...):
```bash
cd /home/group11/final_project_ws
source install/setup.bash
ros2 run vision clip_classifier
```

**Custom labels**:
```bash
ros2 run vision clip_classifier --labels "cat,dog,car,airplane"
```

### 2. Classify Current Image

```bash
ros2 service call /vision/classify_image std_srvs/srv/Trigger
```

### 3. Test Script

```bash
/home/group11/final_project_ws/src/vision/test_clip_classifier.sh
```

---

## 📊 JSON Output Format

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
      { "label": "cat", "confidence": 0.05 }
    ],
    "embedding": {
      "image_vector": [0.123, -0.512, ...],  // 512-dim
      "text_vectors": [
        {"label": "dog", "vector": [0.125, ...]}
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

---

## 🔧 Quick Extractions

### Get Top Prediction
```bash
ros2 service call /vision/classify_image std_srvs/srv/Trigger | \
  grep -A 1000 'message:' | sed '1d' | \
  jq '.output.top_prediction'
```

### Get All Predictions
```bash
ros2 service call /vision/classify_image std_srvs/srv/Trigger | \
  grep -A 1000 'message:' | sed '1d' | \
  jq '.output.all_predictions'
```

### Get Image Embedding
```bash
ros2 service call /vision/classify_image std_srvs/srv/Trigger | \
  grep -A 1000 'message:' | sed '1d' | \
  jq '.output.embedding.image_vector'
```

### Get Processing Time
```bash
ros2 service call /vision/classify_image std_srvs/srv/Trigger | \
  grep -A 1000 'message:' | sed '1d' | \
  jq '.output.metadata.processing_time_ms'
```

---

## 📦 Requirements

```bash
pip install torch transformers pillow
```

**For GPU acceleration**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 👁️ OpenCV Window

Shows:
- Live camera feed
- Top prediction (large, bottom-left)
- Confidence percentage
- Top 3 predictions (right side)

---

## 🎯 Common Use Cases

### Object Recognition
```bash
ros2 run vision clip_classifier --labels "box,cylinder,sphere,cone"
```

### Tool Identification
```bash
ros2 run vision clip_classifier --labels "hammer,screwdriver,wrench,pliers"
```

### Scene Classification
```bash
ros2 run vision clip_classifier --labels "kitchen,office,warehouse,factory"
```

### Quality Control
```bash
ros2 run vision clip_classifier --labels "defective,good,damaged"
```

---

## 📂 Files

- **Node**: `/src/vision/vision/clip_classifier.py`
- **Test**: `/src/vision/test_clip_classifier.sh`
- **Docs**: `/src/vision/CLIP_CLASSIFIER.md`
- **Quick Ref**: `/src/vision/CLIP_QUICK_REF.md` (this file)

---

## 🔄 SAM + CLIP Workflow

```bash
# Terminal 1: Run SAM detector
ros2 run vision simple_sam_detector

# Terminal 2: Run CLIP classifier
ros2 run vision clip_classifier

# Terminal 3: Detect objects
ros2 service call /vision/detect_objects std_srvs/srv/Trigger

# Terminal 4: Classify scene
ros2 service call /vision/classify_image std_srvs/srv/Trigger
```

---

## 💡 Pro Tips

1. **First run is slow** - Downloads 600MB model
2. **Use GPU** - Automatically detected if available
3. **Custom labels** - More specific = better results
4. **Embeddings** - Use for image similarity/search
5. **Combine with SAM** - Detect objects, then classify regions

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model size | 600 MB |
| Embedding dim | 512 |
| CPU inference | ~200-300ms |
| GPU inference | ~50-100ms |
| Memory | ~1.5-2 GB |

---

## 🐛 Quick Fixes

### CLIP not available
```bash
pip install torch transformers pillow
```

### Out of memory
```python
# Edit clip_classifier.py, line ~70
self.device = "cpu"  # Force CPU
```

### No camera feed
```bash
ros2 topic hz /camera/image_raw
```

---

**That's it! You're ready to classify images with CLIP! 🎯**
