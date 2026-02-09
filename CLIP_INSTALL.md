# CLIP Installation Guide

The vision package now uses OpenAI's official CLIP implementation instead of Hugging Face transformers.

## Installation Steps

### 1. Activate your virtual environment (REQUIRED!)
```bash
cd ~/final_project_ws
source vision_venv/bin/activate
# Verify: you should see (vision_venv) in your prompt
```

**⚠️ IMPORTANT:** All pip install commands MUST be run inside the virtual environment!

### 2. Uninstall old dependencies (if installed)
```bash
pip uninstall transformers -y
```

### 3. Install setuptools (REQUIRED - fixes freeze/metadata error)
```bash
pip install --upgrade setuptools wheel
```

### 4. Install OpenAI CLIP from GitHub
```bash
pip install git+https://github.com/openai/CLIP.git
```

### 5. Install required dependencies
```bash
pip install torch torchvision ftfy regex pillow
```

### 6. Rebuild the package
```bash
cd ~/final_project_ws
colcon build --packages-select vision
source install/setup.bash
```

## Troubleshooting

### If pip install freezes or shows metadata-generation-failed error:

**Cause:** Missing or outdated setuptools package

**Solution:**
```bash
pip install --upgrade setuptools wheel
pip install git+https://github.com/openai/CLIP.git
```

### Alternative: Install from local clone
If git+ installation still has issues:
```bash
cd /tmp
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install --upgrade setuptools wheel
pip install .
```

## Verify Installation

Test the CLIP installation:
```bash
python3 -c "import clip; print('CLIP version:', clip.__version__ if hasattr(clip, '__version__') else 'installed'); print('Available models:', clip.available_models())"
```

## Changes Made

- Replaced `transformers.CLIPModel` with `clip.load()`
- Changed from Hugging Face's `AutoProcessor` to OpenAI's `clip.tokenize()` and preprocessing
- Fixed dimension mismatch error (1x512 vs 768x1) in find_object service
- Model name changed from "openai/clip-vit-base-patch32" to "ViT-B/32"

## Testing

After installation, test the find_object service:
```bash
ros2 service call /find_object custom_interfaces/srv/FindObjectReal "{label: 'bowl'}"
```
