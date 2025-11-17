# GraspNet Pixel-to-Real Data Flow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Vision Pipeline with GraspNet                   │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: Camera Images
├─ RGB: /camera/image_raw
├─ Depth: /camera/depth/image_raw  
└─ Info: /camera/camera_info

        │
        ▼
┌───────────────────────┐
│   SAM Detector        │  Detects object regions
│   (simple_sam_detector)│
└───────┬───────────────┘
        │ Publishes SAMDetections
        │ with bounding boxes
        ▼
┌───────────────────────┐
│   CLIP Classifier     │  Classifies objects
│   (clip_classifier)   │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    GraspNet Detector                          │
│                  (graspnet_detector.py)                       │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Step 1: Detect Grasp Point in Pixel Coordinates             │
│  ┌─────────────────────────────────────────┐                 │
│  │  • Analyze object contour/bbox          │                 │
│  │  • Find centroid or best grasp point    │                 │
│  │  • Result: (u, v) = (320, 240)          │                 │
│  │    where u = pixel column (x in image)  │                 │
│  │          v = pixel row (y in image)     │                 │
│  └─────────────────────────────────────────┘                 │
│                    │                                          │
│                    ▼                                          │
│  Step 2: Convert to World Coordinates                        │
│  ┌─────────────────────────────────────────┐                 │
│  │  Call: /pixel_to_real service           │◄────────────────┤─ Service
│  │  Request: {u: 320, v: 240}              │                 │  Client
│  │                                         │                 │
│  │  Service performs calibrated transform: │                 │
│  │  • Uses calibration matrix              │                 │
│  │  • Reads depth at (u, v)                │                 │
│  │  • Computes world position              │                 │
│  │                                         │                 │
│  │  Response: {x: 0.5, y: 0.0, z: 0.8}     │                 │
│  └─────────────────────────────────────────┘                 │
│                    │                                          │
│                    ▼                                          │
│  Step 3: Build Grasp Pose                                    │
│  ┌─────────────────────────────────────────┐                 │
│  │  grasp = {                              │                 │
│  │    "position": {                        │                 │
│  │      "x": 0.5,    ◄─ From service       │                 │
│  │      "y": 0.0,    ◄─ From service       │                 │
│  │      "z": 0.8     ◄─ From service       │                 │
│  │    },                                   │                 │
│  │    "orientation": {                     │                 │
│  │      "x": 0.0, "y": 0.0,                │                 │
│  │      "z": 0.383, "w": 0.924             │                 │
│  │    },                                   │                 │
│  │    "quality_score": 0.7,    ◄─ NON-ZERO │                 │
│  │    "grasp_width": 0.05,     ◄─ NON-ZERO │                 │
│  │    "approach_angle": 45.0,  ◄─ NON-ZERO │                 │
│  │    "pixel_location": [320, 240] ◄─ Original              │
│  │  }                                      │                 │
│  └─────────────────────────────────────────┘                 │
└───────┬───────────────────────────────────────────────────────┘
        │
        ▼
OUTPUT: Grasp Poses
├─ Topic: /vision/grasp_poses (PoseStamped)
└─ Service Response: JSON with all grasps
```

## Pixel to World Transformation Detail

```
┌──────────────────────────────────────────────────────────────────┐
│                    /pixel_to_real Service                        │
│                   (pixel_to_real.py)                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Pixel coordinates (u, v)                                │
│  ┌────────────────────────────┐                                 │
│  │  u = 320 (pixel column)    │                                 │
│  │  v = 240 (pixel row)       │                                 │
│  └────────────────────────────┘                                 │
│                │                                                 │
│                ▼                                                 │
│  STEP 1: Calculate offset from origin                           │
│  ┌────────────────────────────────────────┐                     │
│  │  Origin: (u_origin, v_origin) = (320, 500)                  │
│  │  du = u - u_origin = 320 - 320 = 0      │                   │
│  │  dv = v - v_origin = 240 - 500 = -260   │                   │
│  └────────────────────────────────────────┘                     │
│                │                                                 │
│                ▼                                                 │
│  STEP 2: Apply calibrated scaling                               │
│  ┌────────────────────────────────────────┐                     │
│  │  Calibration:                          │                     │
│  │    scale_x = 0.001923 m/pixel          │                     │
│  │    scale_y = 0.002 m/pixel             │                     │
│  │                                        │                     │
│  │  Transform (coordinate mapping):       │                     │
│  │    x = -dv * scale_x                   │                     │
│  │      = -(-260) * 0.001923              │                     │
│  │      = 0.5 meters                      │                     │
│  │                                        │                     │
│  │    y = -du * scale_y                   │                     │
│  │      = -(0) * 0.002                    │                     │
│  │      = 0.0 meters                      │                     │
│  └────────────────────────────────────────┘                     │
│                │                                                 │
│                ▼                                                 │
│  STEP 3: Read depth and compute z                               │
│  ┌────────────────────────────────────────┐                     │
│  │  Read depth at (u=320, v=240)          │                     │
│  │  depth_sensor = 0.8 meters             │                     │
│  │                                        │                     │
│  │  If depth_reference set:               │                     │
│  │    z = z_table + (depth_ref - depth)   │                     │
│  │    z = 0.8 + (0.8 - 0.8) = 0.8        │                     │
│  └────────────────────────────────────────┘                     │
│                │                                                 │
│                ▼                                                 │
│  OUTPUT: World coordinates (x, y, z)                            │
│  ┌────────────────────────────┐                                 │
│  │  x = 0.5 meters            │                                 │
│  │  y = 0.0 meters            │                                 │
│  │  z = 0.8 meters            │                                 │
│  └────────────────────────────┘                                 │
└──────────────────────────────────────────────────────────────────┘
```

## Service Call Flow

```
User/System
    │
    │ ros2 service call /vision/detect_grasp
    ▼
┌─────────────────────────────────┐
│  GraspNet: detect_grasp_callback│
└─────────────┬───────────────────┘
              │
              │ 1. Get bounding boxes
              ▼
┌─────────────────────────────────┐
│  Call /vision/detect_objects    │
│  Returns: [{x1,y1,x2,y2}, ...]  │
└─────────────┬───────────────────┘
              │
              │ 2. For each bbox
              ▼
┌─────────────────────────────────┐
│  Analyze region                 │
│  Find centroid: (u, v)          │
│  Calculate angle, width         │
└─────────────┬───────────────────┘
              │
              │ 3. Convert to world
              ▼
┌─────────────────────────────────┐
│  _convert_pixel_to_world(u, v)  │
│     │                            │
│     └─► Call /pixel_to_real     │
│         Request: {u: 320, v: 240}
│                │                 │
│                └─► Response: {x, y, z}
└─────────────┬───────────────────┘
              │
              │ 4. Build grasp dict
              ▼
┌─────────────────────────────────┐
│  grasp = {                      │
│    position: {x, y, z},         │
│    quality_score: ≥0.1,         │
│    grasp_width: ≥0.02,          │
│    approach_angle: 30°-150°,    │
│    pixel_location: [u, v]       │
│  }                              │
└─────────────┬───────────────────┘
              │
              │ 5. Return all grasps
              ▼
JSON Response with all grasp poses
```

## Coordinate System Mapping

```
IMAGE SPACE (Pixels)          WORLD SPACE (Meters)
        
    u (→)                           y (←)
    ├────────────►                  ◄────────────┤
    │                                            0
  v │  (320, 240)                    (0.5, 0.0, 0.8)
  ( │     ◉ Green Box                      ◉ Green Box
  ↓ │                                      
  ) │                               x       
    │  (320, 500)                   │       
    │     ⊕ Origin                  ├─────► (forward)
    │                               0       
    │                                       
   480                              z       
                                    │       
   640                              ├─────► (height)
                                   0.8      
                                            
Mapping:                                    
  u increases right → y DECREASES           
  v increases down  → x DECREASES           
  depth             → z (table at 0.8m)     
```

## Data Structure

```json
{
  "success": true,
  "total_grasps": 3,
  "grasps": [
    {
      "grasp_id": 0,
      "object_id": "green_box",
      "position": {
        "x": 0.5,     // ← From /pixel_to_real
        "y": 0.0,     // ← From /pixel_to_real  
        "z": 0.8      // ← From /pixel_to_real
      },
      "orientation": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.383,   // Quaternion from angle
        "w": 0.924
      },
      "quality_score": 0.7,      // ✓ NON-ZERO
      "grasp_width": 0.05,       // ✓ NON-ZERO (≥0.02m)
      "approach_angle": 45.0,    // ✓ NON-ZERO (30°-150°)
      "pixel_location": [320, 240],  // Original (u,v)
      "depth_value": 0.8
    }
  ]
}
```

## Key Guarantees

1. **Accurate World Coordinates**
   - Uses calibrated transformation via `/pixel_to_real`
   - Validated against known calibration points
   
2. **Non-Zero Values**
   - `quality_score`: Always in range [0.1, 1.0]
   - `grasp_width`: Always ≥ 0.02m (2cm minimum)
   - `approach_angle`: Always between 30° and 150°

3. **Traceability**
   - Original pixel coordinates stored in `pixel_location`
   - Can verify transformation manually

4. **Robustness**
   - Handles service unavailability (returns defaults)
   - Validates depth readings
   - Clamps values to valid ranges
