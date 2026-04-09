# Vision Data Collector & Excel Exporter

Runs the full SAM → CLIP → Scene Understanding pipeline **once**, stores the result in a rolling
JSON history (last 20 runs), and writes a ready-to-open Excel workbook.

---

## Output Files

| File | Location | Description |
|---|---|---|
| `vision_runs_history.json` | workspace root (next to `README.md`) | Rolling database of last 20 runs |
| `vision_runs_export.xlsx` | workspace root | Excel workbook, 4 sheets |

### Excel Sheets

| Sheet | One row per | Key columns |
|---|---|---|
| **Runs** | Each script invocation | timestamp, total objects, graspable, avg confidence, stability %, scene description, latency |
| **Objects** | Each detected object | CLIP label, bbox coords, SAM + CLIP confidence, distance cm, IoU score, grasp quality |
| **Relations** | Each spatial relation | subject → relation → target, confidence, 2D distance |
| **Grasps** | Each grasp pose | x/y/z position (m), quality score, gripper width (m) |

---

## Prerequisites

### 1. Install openpyxl (one-time)

```bash
pip install openpyxl
```

### 2. ROS2 workspace must be built

```bash
cd ~/ros2_ws   # or wherever your workspace is
colcon build --packages-select vision
source install/setup.bash
```

---

## How to Run

### Step 1 — Start the vision nodes (separate terminals)

```bash
# Terminal 1: SAM detector (required)
ros2 run vision simple_sam_detector

# Terminal 2: CLIP classifier (required for object labels)
ros2 run vision clip_classifier

# Terminal 3: Scene understanding (required for relations + grasp data)
ros2 run vision scene_understanding

# Terminal 4: GraspNet detector (optional — needed for grasp positions)
ros2 run vision graspnet_detector
```

Wait until you see the nodes announce their services in the terminal output before continuing.

### Step 2 — Make sure a camera is publishing

```bash
# Simulation (Gazebo):
ros2 launch ur_yt_sim spawn_ur5_camera_gripper_moveit.launch.py

# Real RealSense camera:
ros2 launch realsense2_camera rs_launch.py

# Static image (for testing without hardware):
ros2 launch vision vision_with_camera.launch.py camera_type:=file image_file:=path/to/image.jpg
```

### Step 3 — Run the collector

```bash
# Option A: directly with python3
python3 vision_scripts/collect_and_export.py

# Option B: as a ROS2 entry point (after colcon build)
ros2 run vision collect_and_export
```

### Expected terminal output

```
============================================================
  Vision Pipeline Data Collector & Excel Exporter
============================================================
  History file : /path/to/vision/vision_runs_history.json
  Excel file   : /path/to/vision/vision_runs_export.xlsx
  Max runs kept: 20
============================================================
Waiting for /vision/run_pipeline ...
RUN #1 — Step 1: SAM /vision/run_pipeline
  SAM: 3 objects detected
RUN #1 — Step 2: CLIP /vision/classify_bbox_filtered
  CLIP: 2 regions classified
RUN #1 — Step 3: Scene /vision/understand_scene
  Scene: 3 objects, 4 relations
RUN #1 complete in 2.341s — 3 objects, 4 relations
============================================================
[OK] History saved → .../vision_runs_history.json  (1 runs stored)
[OK] Excel exported → .../vision_runs_export.xlsx

Done. Open vision_runs_export.xlsx to view results.
```

---

## Run it multiple times

Each time you run the script it **appends** one new entry to the JSON and **overwrites** the Excel.
After 20 runs the oldest entry is automatically dropped.

```bash
# Run 5 times in a loop (bash)
for i in {1..5}; do
    python3 vision_scripts/collect_and_export.py
    sleep 2
done
```

---

## Debugging

### Problem: "service not available — skipping"

The script will skip any service that does not respond within 5 seconds and
continue with the remaining services without crashing.

**Fix:** Check that the required node is running:

```bash
ros2 node list           # should show /simple_sam_detector, /clip_classifier, etc.
ros2 service list        # should show /vision/run_pipeline, /vision/classify_bbox_filtered, etc.
```

If a node is missing, start it (see Step 1 above).

---

### Problem: No camera image / SAM returns 0 objects

```bash
# Check camera topics are publishing
ros2 topic list | grep camera
ros2 topic hz /camera/image_raw       # should show ~30 Hz

# Check SAM subscriber got a frame
ros2 topic echo /vision/status        # shows uptime and detection count
```

SAM only detects when a frame has been received. Wait a second after starting the camera before calling the collector.

---

### Problem: CLIP returns "No classified regions"

CLIP auto-classifies when it receives a message on `/vision/sam_detections`.
The collector calls `/vision/run_pipeline` first (which publishes to that topic), then
waits 500 ms before calling CLIP. If CLIP is slow on first run (model loading):

```bash
# Call run_pipeline manually first to warm up CLIP
ros2 service call /vision/run_pipeline std_srvs/srv/Trigger

# Wait a few seconds, then run the collector
python3 vision_scripts/collect_and_export.py
```

---

### Problem: Scene understanding times out

Scene understanding calls `/vision/detect_objects` internally which can take a few
seconds. The collector allows **15 seconds** for it. If it still times out:

```bash
# Test scene understanding manually
ros2 service call /vision/understand_scene std_srvs/srv/Trigger
```

If that also hangs, check that `simple_sam_detector` is running (scene understanding
calls it internally).

---

### Problem: "openpyxl not installed"

```bash
pip install openpyxl

# Verify:
python3 -c "import openpyxl; print(openpyxl.__version__)"
```

The script will still save `vision_runs_history.json` even without openpyxl — only
the Excel export is skipped.

---

### Problem: Excel file is open in Excel and the script fails to overwrite it

Close the Excel file first, then re-run the collector.

---

### Inspect the raw JSON history

```bash
# Pretty-print the history file
python3 -c "import json; print(json.dumps(json.load(open('vision_runs_history.json')), indent=2))"

# Show just the run summaries
python3 -c "
import json
runs = json.load(open('vision_runs_history.json'))
for r in runs:
    m = r['meta']; s = r['sam']; sc = r['scene']
    print(f\"Run #{m['run_no']} | {m['timestamp']} | objects={s.get('total_detections','?')} | latency={m['latency_s']}s\")
"
```

---

### Manually clear the history

```bash
rm vision_runs_history.json
```

The next run will start fresh from run #1.

---

## File Structure

```
vision/
├── vision_scripts/
│   └── collect_and_export.py   ← the collector script
├── vision_runs_history.json    ← created on first run
├── vision_runs_export.xlsx     ← created on first run
└── docs/
    └── COLLECT_AND_EXPORT.md   ← this file
```

---

## Services Called (in order)

| # | Service | Type | Purpose |
|---|---|---|---|
| 1 | `/vision/run_pipeline` | `std_srvs/Trigger` | SAM detects objects, publishes to topic |
| 2 | `/vision/classify_bbox_filtered` | `std_srvs/Trigger` | CLIP labels each region (conf > 0.5) |
| 3 | `/vision/understand_scene` | `std_srvs/Trigger` | Full scene: objects + relations + grasps |

If a service is not running the step is skipped and that section of the run record will show `"success": false`.
The JSON and Excel are still saved with whatever data was collected.
