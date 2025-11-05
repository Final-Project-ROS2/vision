# Visualization & Terminal Output Added to SAM+CLIP Pipeline

## Summary of Changes

Added comprehensive visualization and terminal output capabilities to `sam_clip_pipeline.py`.

## New Features

### 1. **OpenCV Window Visualization**
- Creates window: "SAM + CLIP Pipeline Results"
- Size: 1000x750 pixels
- Shows detected objects with:
  - Color-coded bounding boxes
  - Semi-transparent segmentation masks
  - Object labels and IDs
  - CLIP confidence scores
  - SAM confidence scores
  - Legend at bottom

### 2. **Terminal JSON Output**
- Prints complete JSON results to terminal after processing
- Easy to copy/paste or redirect to file
- Full classification details included

### 3. **Saved Visualization**
- Automatically saves annotated image to: `~/sam_clip_outputs/visualization_YYYYMMDD_HHMMSS.jpg`
- Includes all annotations and labels

## Visualization Details

### Color-Coded Objects
Each detected object gets a unique color:
- Green, Blue, Red, Cyan, Magenta, Yellow, Light Green, Orange
- Colors cycle if more than 8 objects

### Labels Include:
- **Object ID**: `#0`, `#1`, etc.
- **Class Label**: From CLIP classification
- **CLIP Confidence**: Classification confidence
- **SAM Confidence**: Detection confidence

### Title Bar
Shows:
- Pipeline name
- Frame number
- Total objects detected

### Legend
Bottom-left corner shows:
- First 5 objects with color and classification

## Usage

```bash
# Terminal 1: Start the pipeline
ros2 run vision sam_clip_pipeline

# Terminal 2: Process scene
ros2 service call /vision/process_pipeline std_srvs/srv/Trigger
```

## Output Example

### Terminal Output:
```
================================================================================
📋 JSON OUTPUT:
================================================================================
{
  "pipeline": "sam_clip_integrated",
  "success": true,
  "model": "openai/clip-vit-base-patch32",
  "input": {
    "image_id": "frame_000001",
    "image_shape": [480, 640, 3],
    "num_regions": 3,
    "candidate_labels": ["cobot", "green_cube", "drill", ...]
  },
  "output": {
    "classified_regions": [
      {
        "region_id": 0,
        "bbox": [100, 150, 200, 250],
        "sam_confidence": 0.85,
        "area": 10000,
        "top_prediction": {
          "label": "drill",
          "confidence": 0.92
        },
        "all_predictions": [...]
      }
    ],
    "summary": {
      "total_regions": 3,
      "processing_time_ms": 1250
    },
    "json_file": "/home/user/sam_clip_outputs/sam_clip_results_*.json"
  },
  "metadata": {
    "timestamp": "2025-11-04T12:00:00.000000Z",
    "device": "cuda",
    "output_directory": "/home/user/sam_clip_outputs"
  }
}
================================================================================
✅ PIPELINE COMPLETE
   Detected: 3 objects
   Classified: 3 regions
   JSON saved to: /home/user/sam_clip_outputs/sam_clip_results_20251104_120000.json
   Visualization displayed in window: 'SAM + CLIP Pipeline Results'
================================================================================
```

### OpenCV Window:
- Shows annotated image with all detections
- Color-coded bounding boxes and masks
- Labels with confidence scores
- Legend at bottom
- Press any key to close

### Saved Files:
1. **JSON Results**: `~/sam_clip_outputs/sam_clip_results_YYYYMMDD_HHMMSS.json`
2. **Visualization**: `~/sam_clip_outputs/visualization_YYYYMMDD_HHMMSS.jpg`

## Benefits

1. **Visual Feedback**: See exactly what was detected and classified
2. **Easy Debug**: Visual inspection of detection quality
3. **Terminal Copy**: Easy to copy JSON output from terminal
4. **Permanent Record**: Both JSON and image saved to disk
5. **Color Coding**: Quick identification of multiple objects
6. **Dual Confidence**: See both SAM and CLIP confidence scores

## Tips

### Save Terminal Output to File:
```bash
ros2 service call /vision/process_pipeline std_srvs/srv/Trigger 2>&1 | tee pipeline_output.log
```

### Extract JSON from Terminal:
Look for the section between:
```
📋 JSON OUTPUT:
================================================================================
{ ... }
================================================================================
```

### Close Visualization:
- Press any key while the OpenCV window is focused
- Or let it stay open for review

## Troubleshooting

### Window Not Showing:
- Ensure X11 forwarding is enabled if using SSH
- Check: `echo $DISPLAY`
- Try: `export DISPLAY=:0`

### Window Too Small/Large:
- Edit window size in code: `cv2.resizeWindow(self.window_name, WIDTH, HEIGHT)`

### Can't Read Labels:
- Increase font size in `_display_results()` method
- Adjust label background size
