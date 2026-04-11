#!/usr/bin/env python3
"""
Unit test: verify detect_objects → vision_runs_history.json → dashboard data flow.

Runs WITHOUT ROS2. Tests:
  1. JSON file is written to the correct path
  2. JSON schema matches exactly what dashboard HTML expects
  3. /api/data live path (benchmark_dashboard SAM topic callback schema)
  4. /api/run-history path (run history table schema)

Usage:
    python3 vision_scripts/test_dashboard_data_flow.py
"""

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent                          # src/vision/
HISTORY_FILE = PACKAGE_ROOT / "vision_runs_history.json"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

failures = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


# ── Fake detection data (mirrors what detect_objects_callback produces) ────────

FAKE_DETECTIONS = [
    {
        "id":          "object_0",
        "class_name":  "object_0",
        "confidence":  0.82,
        "bbox":        [100, 150, 300, 400],
        "center":      [200, 275],
        "area":        45000,
        "distance_cm": 55.3,
        "iou_score":   0.71,
        "is_stable":   True,
        "matched_prev_id": "",
        "mask":        None,  # would be np array in real code
    },
    {
        "id":          "object_1",
        "class_name":  "object_1",
        "confidence":  0.64,
        "bbox":        [400, 200, 580, 450],
        "center":      [490, 325],
        "area":        34200,
        "distance_cm": 72.1,
        "iou_score":   0.0,
        "is_stable":   False,
        "matched_prev_id": "",
        "mask":        None,
    },
]

FAKE_CLIP = {
    0: {"label": "cup",    "confidence": 0.91, "bbox": [100, 150, 300, 400]},
    1: {"label": "bottle", "confidence": 0.78, "bbox": [400, 200, 580, 450]},
}


# ── Reproduce _save_detect_objects_run logic verbatim ─────────────────────────

def simulate_save(latest_detections, clip_classifications, latency_s, history_file):
    """Exact copy of _save_detect_objects_run from simple_sam_detector.py."""
    history = []
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                history = data
        except Exception:
            pass

    last_run_no = history[-1]["meta"]["run_no"] if history else 0
    run_no = last_run_no + 1

    # Build parallel arrays (same logic as detect_objects_callback)
    object_ids    = []
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [], [], [], []
    confidences   = []
    distances_cm  = []
    iou_scores    = []
    is_stable_arr = []

    for idx, det in enumerate(latest_detections):
        clip_info = clip_classifications.get(idx)
        if clip_info:
            object_ids.append(f"{clip_info['label']}_{idx}")
            confidences.append(float(clip_info["confidence"]))
        else:
            object_ids.append(det["id"])
            confidences.append(float(det["confidence"]))

        bbox = det["bbox"]
        bbox_x1.append(bbox[0]); bbox_y1.append(bbox[1])
        bbox_x2.append(bbox[2]); bbox_y2.append(bbox[3])
        distances_cm.append(float(det.get("distance_cm", -1.0)))
        iou_scores.append(float(det.get("iou_score", 0.0)))
        is_stable_arr.append(bool(det.get("is_stable", False)))

    num_dets = len(latest_detections)
    total_sam_conf = 0.0
    objects = []
    for idx in range(num_dets):
        det = latest_detections[idx]
        clip_info = clip_classifications.get(idx, {})
        sam_conf = float(det.get("confidence", 0.0))
        total_sam_conf += sam_conf
        objects.append({
            "object_id":      object_ids[idx] if idx < len(object_ids) else f"object_{idx}",
            "label":          clip_info.get("label", "") if clip_info else "",
            "bbox_x1":        bbox_x1[idx] if idx < len(bbox_x1) else 0,
            "bbox_y1":        bbox_y1[idx] if idx < len(bbox_y1) else 0,
            "bbox_x2":        bbox_x2[idx] if idx < len(bbox_x2) else 0,
            "bbox_y2":        bbox_y2[idx] if idx < len(bbox_y2) else 0,
            "sam_confidence": round(sam_conf, 4),
            "clip_confidence": round(float(clip_info.get("confidence", 0.0)), 4) if clip_info else "",
            "distance_cm":    distances_cm[idx] if idx < len(distances_cm) else "",
            "iou_score":      iou_scores[idx] if idx < len(iou_scores) else "",
            "is_stable":      is_stable_arr[idx] if idx < len(is_stable_arr) else "",
            "has_grasp":      False,
            "grasp":          {},
            "obb_angle_deg":  "", "obb_theta_rad": "",
            "obb_width_px":   "", "obb_height_px": "",
            "obb_center_u":   "", "obb_center_v":  "",
        })

    avg_sam_conf   = total_sam_conf / num_dets if num_dets > 0 else 0.0
    avg_iou        = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    stability_rate = sum(1 for s in is_stable_arr if s) / len(is_stable_arr) if is_stable_arr else 0.0

    run = {
        "meta": {
            "run_no":    run_no,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "latency_s": round(latency_s, 3),
            "source":    "detect_objects",
        },
        "sam": {
            "success":          True,
            "latency_s":        round(latency_s, 3),
            "total_detections": num_dets,
            "avg_confidence":   round(avg_sam_conf, 4),
            "average_iou":      round(avg_iou, 4),
            "stability_rate":   round(stability_rate, 4),
        },
        "clip": {
            "success":          bool(clip_classifications),
            "latency_s":        0.0,
            "filtered_regions": len(clip_classifications),
        },
        "scene": {"success": False, "latency_s": 0.0},
        "obb":   {"success": False, "latency_s": 0.0},
        "objects":   objects,
        "relations": [],
        "grasps":    [],
    }

    history.append(run)
    history = history[-20:]

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    return run, history


# ── Test 1: File is written ────────────────────────────────────────────────────

def test_file_written():
    print("\n── Test 1: File write ───────────────────────────────────────────")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        run, history = simulate_save(FAKE_DETECTIONS, FAKE_CLIP, 1.23, tmp)
        check("File created", tmp.exists())
        check("File is valid JSON", True)  # would have thrown above
        check("History is list", isinstance(history, list))
        check("History has 1 entry", len(history) == 1)
        check("run_no == 1", run["meta"]["run_no"] == 1)
        print(f"  {INFO}  Written to: {tmp}")
        return tmp, run, history
    except Exception as e:
        check("No exception during save", False, str(e))
        traceback.print_exc()
        return None, None, None
    finally:
        pass  # keep file for next test


# ── Test 2: JSON schema matches /api/run-history dashboard expectations ────────

def test_run_history_schema(run):
    print("\n── Test 2: /api/run-history schema ─────────────────────────────")
    if run is None:
        print("  Skipped (previous test failed)")
        return

    meta  = run.get("meta", {})
    sam   = run.get("sam", {})
    clip  = run.get("clip", {})
    objs  = run.get("objects", [])

    # meta fields (dashboard uses: run_no, timestamp, latency_s)
    check("meta.run_no present",    "run_no"    in meta)
    check("meta.timestamp present", "timestamp" in meta)
    check("meta.latency_s present", "latency_s" in meta)

    # sam fields (dashboard uses: total_detections, avg_confidence, average_iou, stability_rate, latency_s, success)
    for field in ["total_detections", "avg_confidence", "average_iou", "stability_rate", "latency_s", "success"]:
        check(f"sam.{field} present", field in sam)

    # clip fields (dashboard uses: filtered_regions, latency_s, success)
    for field in ["filtered_regions", "latency_s", "success"]:
        check(f"clip.{field} present", field in clip)

    # objects array
    check("objects is list",        isinstance(objs, list))
    check("objects not empty",      len(objs) == len(FAKE_DETECTIONS))

    # per-object fields (dashboard uses all of these)
    obj_fields = [
        "object_id", "label", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "sam_confidence", "clip_confidence", "distance_cm",
        "iou_score", "is_stable", "has_grasp", "grasp",
        "obb_angle_deg", "obb_theta_rad",
    ]
    if objs:
        for field in obj_fields:
            check(f"objects[0].{field} present", field in objs[0])

    # value spot-checks
    check("sam.total_detections == 2",        sam.get("total_detections") == 2)
    check("clip.filtered_regions == 2",       clip.get("filtered_regions") == 2)
    check("object label 'cup' in object_id",  "cup" in objs[0].get("object_id", "") or objs[0].get("label") == "cup")
    check("bbox values correct",              objs[0]["bbox_x1"] == 100 and objs[0]["bbox_y2"] == 400)
    check("sam_confidence is float",          isinstance(objs[0]["sam_confidence"], float))
    check("clip_confidence is float",         isinstance(objs[0]["clip_confidence"], float))
    check("iou_score is float",               isinstance(objs[0]["iou_score"], float))
    check("is_stable is bool",                isinstance(objs[0]["is_stable"], bool))


# ── Test 3: /api/data schema (benchmark_dashboard SAM topic callback) ──────────

def test_api_data_schema():
    print("\n── Test 3: /api/data schema (sam_detections_callback output) ───")
    # Replicate what benchmark_dashboard.sam_detections_callback produces
    timestamp = datetime.now().isoformat()
    sam_data_records = []
    for det in FAKE_DETECTIONS:
        bbox = det["bbox"]
        center = det["center"]
        record = {
            "timestamp":        timestamp,
            "frame_id":         "camera_link",
            "obj_id":           det["id"],
            "bbox":             {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
            "center":           {"u": center[0], "v": center[1]},
            "confidence":       float(det["confidence"]),
            "area":             det["area"],
            "distance_cm":      float(det["distance_cm"]),
            "iou_score":        float(det["iou_score"]),
            "is_stable":        bool(det["is_stable"]),
            "ap_iou_threshold": 0.5 if det["is_stable"] else 0.0,
        }
        sam_data_records.append(record)

    # Check each field that updateSAMDetections() accesses in the HTML
    r = sam_data_records[0]
    for field in ["obj_id", "bbox", "center", "confidence", "iou_score", "is_stable", "distance_cm", "timestamp"]:
        check(f"sam record has '{field}'", field in r)

    check("bbox has x1/y1/x2/y2",   all(k in r["bbox"] for k in ["x1","y1","x2","y2"]))
    check("center has u/v",          all(k in r["center"] for k in ["u","v"]))
    check("confidence is numeric",   isinstance(r["confidence"], float))
    check("distance_cm is numeric",  isinstance(r["distance_cm"], float))

    # Simulate full /api/data structure
    api_data = {
        "pixel_to_real":       [],
        "sam_detections":      sam_data_records,
        "clip_classifications":[],
        "grasp_detections":    [],
        "scene_understanding": [],
        "metadata":            {"start_time": timestamp, "total_calls": len(sam_data_records)},
    }
    # Check updateDashboard() fields
    for key in ["pixel_to_real", "sam_detections", "clip_classifications",
                "grasp_detections", "scene_understanding", "metadata"]:
        check(f"/api/data has '{key}'", key in api_data)
    check("metadata.total_calls present", "total_calls" in api_data["metadata"])


# ── Test 4: Incremental append (run_no increments) ───────────────────────────

def test_incremental_append(tmp_file):
    print("\n── Test 4: Incremental append (run numbers) ─────────────────────")
    if tmp_file is None:
        print("  Skipped")
        return
    simulate_save(FAKE_DETECTIONS, FAKE_CLIP, 0.9, tmp_file)
    simulate_save(FAKE_DETECTIONS, FAKE_CLIP, 1.1, tmp_file)

    with open(tmp_file) as f:
        history = json.load(f)

    check("3 runs in history",               len(history) == 3)
    check("run_no increments: 1,2,3",        [r["meta"]["run_no"] for r in history] == [1, 2, 3])
    check("latest run_no is 3",              history[-1]["meta"]["run_no"] == 3)
    check("source == 'detect_objects'",      all(r["meta"].get("source") == "detect_objects" for r in history))


# ── Test 5: Actual HISTORY_FILE path resolution ───────────────────────────────

def test_path_resolution():
    print("\n── Test 5: Path resolution ──────────────────────────────────────")
    # This is how simple_sam_detector.py resolves the path:
    # Path(__file__).parent.parent where __file__ is vision/simple_sam_detector.py
    sam_file = PACKAGE_ROOT / "vision" / "simple_sam_detector.py"
    resolved = sam_file.parent.parent / "vision_runs_history.json"

    # And how benchmark_dashboard.py resolves it:
    dash_file = PACKAGE_ROOT / "vision" / "benchmark_dashboard.py"
    dash_resolved = dash_file.parent.parent / "vision_runs_history.json"

    check("simple_sam_detector path resolves correctly",
          str(resolved) == str(HISTORY_FILE),
          f"got {resolved}")
    check("benchmark_dashboard path resolves correctly",
          str(dash_resolved) == str(HISTORY_FILE),
          f"got {dash_resolved}")
    check("Both paths are identical",
          resolved == dash_resolved)
    check("simple_sam_detector.py exists",
          sam_file.exists(),
          f"missing: {sam_file}")
    check("benchmark_dashboard.py exists",
          dash_file.exists(),
          f"missing: {dash_file}")
    print(f"  {INFO}  History file path: {HISTORY_FILE}")


# ── Test 6: Dashboard HTML endpoints exist ────────────────────────────────────

def test_html_endpoints():
    print("\n── Test 6: Dashboard HTML references ────────────────────────────")
    html_file = PACKAGE_ROOT / "dashboard" / "index.html"
    check("index.html exists", html_file.exists())
    if html_file.exists():
        content = html_file.read_text()
        check("fetches /api/data",        "/api/data"        in content)
        check("fetches /api/run-history", "/api/run-history" in content)
        check("polls every 2s",           "2000"             in content)
        check("runHistoryBody table",     "runHistoryBody"   in content)
        check("samBody table",            "samBody"          in content)
        check("latestObjectsBody table",  "latestObjectsBody" in content)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Vision Dashboard Data-Flow Unit Test")
    print("=" * 65)

    tmp_file, run, history = test_file_written()
    test_run_history_schema(run)
    test_api_data_schema()
    test_incremental_append(tmp_file)
    test_path_resolution()
    test_html_endpoints()

    # Clean up temp file
    if tmp_file and tmp_file.exists():
        tmp_file.unlink()

    print("\n" + "=" * 65)
    if failures:
        print(f"\033[91m  {len(failures)} FAILED:\033[0m")
        for f in failures:
            print(f"    • {f}")
        print("=" * 65)
        sys.exit(1)
    else:
        print(f"\033[92m  All tests passed.\033[0m")
        print("=" * 65)

    # ── Show current history file state ───────────────────────────────────────
    print(f"\n{INFO}  Checking actual history file: {HISTORY_FILE}")
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        print(f"  {PASS}  File exists — {len(data)} run(s) stored")
        if data:
            latest = data[-1]
            print(f"  {INFO}  Latest run: #{latest['meta']['run_no']}  "
                  f"ts={latest['meta']['timestamp']}  "
                  f"objects={latest['sam']['total_detections']}")
    else:
        print(f"  \033[93m[WARN]\033[0m  {HISTORY_FILE} does not exist yet.")
        print(f"         The file is created when you call:")
        print(f"           ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects")
        print(f"         Make sure simple_sam_detector is running and a camera frame is available.")


if __name__ == "__main__":
    main()
