#!/home/group11/vision_venv/bin/python3
"""
Benchmark Dashboard Node

Monitors all vision service calls and publishes benchmark data for visualization.
Does NOT modify any existing service nodes - only monitors and logs data.

Provides HTTP server for real-time dashboard viewing at http://localhost:8080

Services:
    /benchmark/clear_data - Clear all benchmark data
    
Topics Published:
    /benchmark/data - JSON string of all benchmark data

Usage:
    ros2 run vision benchmark_dashboard
    

    ros2 service call /benchmark/clear_data std_srvs/srv/Trigger


    Then open browser: http://localhost:8080
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger
from std_msgs.msg import String
import json
import numpy as np
import time
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import os
from pathlib import Path

# Import custom interfaces
try:
    from custom_interfaces.srv import (
        PixelToReal, DetectObjects, ClassifyBBox, 
        DetectGrasps, UnderstandScene, DetectGraspBBox
    )
    from custom_interfaces.msg import SAMDetections, SceneUnderstanding
    CUSTOM_INTERFACES_AVAILABLE = True
except ImportError:
    CUSTOM_INTERFACES_AVAILABLE = False
    print("Custom interfaces not available. Build custom_interfaces package first.")


class _ROSJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles ROS/numpy integer and float types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class BenchmarkDashboard(Node):
    """
    Benchmark Dashboard - Monitors vision services and provides web interface
    """
    
    def __init__(self):
        super().__init__('benchmark_dashboard')
        
        # Callback group for service calls
        self.callback_group = ReentrantCallbackGroup()
        
        # Benchmark data storage
        self.data = {
            'pixel_to_real': [],
            'sam_detections': [],
            'clip_classifications': [],
            'grasp_detections': [],
            'obb_detections': [],
            'scene_understanding': [],
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'total_calls': 0
            }
        }
        
        # Service clients for monitoring (we'll call and record)
        if CUSTOM_INTERFACES_AVAILABLE:
            self.pixel_to_real_client = self.create_client(
                PixelToReal, 
                '/pixel_to_real',
                callback_group=self.callback_group
            )
            
            self.detect_objects_client = self.create_client(
                DetectObjects,
                '/vision/detect_objects',
                callback_group=self.callback_group
            )
            
            self.classify_bbox_client = self.create_client(
                ClassifyBBox,
                '/vision/classify_bb',
                callback_group=self.callback_group
            )
            
            self.detect_grasps_client = self.create_client(
                DetectGrasps,
                '/vision/detect_grasp',
                callback_group=self.callback_group
            )
            
            self.understand_scene_client = self.create_client(
                UnderstandScene,
                '/vision/understand_scene',
                callback_group=self.callback_group
            )
        
        # Subscriber to SAM detections topic for automatic monitoring
        self.sam_sub = self.create_subscription(
            SAMDetections,
            '/vision/sam_detections',
            self.sam_detections_callback,
            10
        )
        
        # Subscriber to scene understanding topic
        self.scene_sub = self.create_subscription(
            SceneUnderstanding,
            '/vision/scene_understanding',
            self.scene_understanding_callback,
            10
        )
        
        # Publisher for benchmark data (other nodes can subscribe)
        self.data_publisher = self.create_publisher(
            String,
            '/benchmark/data',
            10
        )
        
        # Service to clear benchmark data
        self.clear_service = self.create_service(
            Trigger,
            '/benchmark/clear_data',
            self.clear_data_callback,
            callback_group=self.callback_group
        )
        
        # Timer to publish data periodically
        self.publish_timer = self.create_timer(1.0, self.publish_data)

        # Timer to sync CLIP/OBB/GraspNet/Pixel-to-Real from vision_runs_history.json
        self.sync_timer = self.create_timer(2.0, self.sync_from_run_history)

        # Start HTTP server in separate thread
        self.start_http_server()
        
        self.get_logger().info('Benchmark Dashboard started')
        self.get_logger().info('Dashboard available at http://localhost:8080')
        self.get_logger().info('Monitoring vision services...')
    
    def sam_detections_callback(self, msg):
        """Monitor SAM detections from topic"""
        timestamp = datetime.now().isoformat()
        
        for detection in msg.detections:
            sam_data = {
                'timestamp': timestamp,
                'frame_id': msg.header.frame_id,
                'obj_id': detection.object_id,
                'bbox': {
                    'x1': int(detection.bbox[0]),
                    'y1': int(detection.bbox[1]),
                    'x2': int(detection.bbox[2]),
                    'y2': int(detection.bbox[3])
                },
                'center': {
                    'u': int(detection.center[0]),
                    'v': int(detection.center[1])
                },
                'confidence': float(detection.confidence),
                'area': int(detection.area),
                'distance_cm': float(detection.distance_cm),
                'iou_score': float(detection.iou_score),
                'is_stable': bool(detection.is_stable_detection),
                'ap_iou_threshold': 0.5 if detection.is_stable_detection else 0.0
            }
            
            self.data['sam_detections'].append(sam_data)
        
        # Limit data size (keep last 1000 records)
        if len(self.data['sam_detections']) > 1000:
            self.data['sam_detections'] = self.data['sam_detections'][-1000:]
        
        self.data['metadata']['total_calls'] += 1
    
    def scene_understanding_callback(self, msg):
        """Monitor scene understanding from topic"""
        timestamp = datetime.now().isoformat()
        
        relations = []
        for rel in msg.all_relations:
            relations.append({
                'subject': rel.subject_label,
                'relation': rel.relation,
                'object': rel.object_label,
                'confidence': float(rel.confidence)
            })
        
        scene_data = {
            'timestamp': timestamp,
            'scene_id': msg.scene_id,
            'total_objects': int(msg.total_objects),
            'relations': relations,
            'object_labels': list(msg.object_labels),
            'object_counts': [int(c) for c in msg.object_counts],
            'graspable_objects': msg.graspable_objects,
            'average_distance_cm': float(msg.average_distance_cm),
            'scene_description': msg.scene_description,
            'spatial_accuracy': self.calculate_spatial_accuracy(relations),
            'adjacency_accuracy': self.calculate_adjacency_accuracy(relations)
        }
        
        self.data['scene_understanding'].append(scene_data)
        
        # Limit data size
        if len(self.data['scene_understanding']) > 100:
            self.data['scene_understanding'] = self.data['scene_understanding'][-100:]
        
        self.data['metadata']['total_calls'] += 1
    
    def calculate_spatial_accuracy(self, relations):
        """Calculate spatial relationship accuracy (placeholder - needs ground truth)"""
        # For now, return confidence-based metric
        if not relations:
            return 0.0
        avg_confidence = sum(r['confidence'] for r in relations) / len(relations)
        return round(avg_confidence * 100, 2)
    
    def calculate_adjacency_accuracy(self, relations):
        """Calculate adjacency accuracy (near/touching relations)"""
        adjacency_relations = [r for r in relations if r['relation'] in ['near', 'touching']]
        if not adjacency_relations:
            return 0.0
        avg_confidence = sum(r['confidence'] for r in adjacency_relations) / len(adjacency_relations)
        return round(avg_confidence * 100, 2)
    
    def add_pixel_to_real_record(self, u, v, x, y, z):
        """Add pixel to real conversion record"""
        timestamp = datetime.now().isoformat()
        
        record = {
            'timestamp': timestamp,
            'test_id': len(self.data['pixel_to_real']) + 1,
            'input': {'u': u, 'v': v},
            'output': {'x': float(x), 'y': float(y), 'z': float(z)}
        }
        
        self.data['pixel_to_real'].append(record)
        
        # Limit data size
        if len(self.data['pixel_to_real']) > 1000:
            self.data['pixel_to_real'] = self.data['pixel_to_real'][-1000:]
        
        self.data['metadata']['total_calls'] += 1
    
    def add_clip_classification_record(self, bbox, label, confidence, is_correct=None):
        """Add CLIP classification record"""
        timestamp = datetime.now().isoformat()
        
        record = {
            'timestamp': timestamp,
            'test_id': len(self.data['clip_classifications']) + 1,
            'bbox': bbox,
            'label': label,
            'confidence': float(confidence),
            'top1_accuracy': is_correct  # True/False/None if unknown
        }
        
        self.data['clip_classifications'].append(record)
        
        # Limit data size
        if len(self.data['clip_classifications']) > 1000:
            self.data['clip_classifications'] = self.data['clip_classifications'][-1000:]
        
        self.data['metadata']['total_calls'] += 1
    
    def add_grasp_detection_record(self, grasp_pose):
        """Add grasp detection record"""
        timestamp = datetime.now().isoformat()
        
        record = {
            'timestamp': timestamp,
            'test_id': len(self.data['grasp_detections']) + 1,
            'object_id': grasp_pose.object_id,
            'bbox': [int(v) for v in grasp_pose.bbox],
            'pixel_position': {
                'u': int(grasp_pose.bbox[0]) + (int(grasp_pose.bbox[2]) - int(grasp_pose.bbox[0])) // 2,
                'v': int(grasp_pose.bbox[1]) + (int(grasp_pose.bbox[3]) - int(grasp_pose.bbox[1])) // 2
            },
            'world_position': {
                'x': float(grasp_pose.position.x),
                'y': float(grasp_pose.position.y),
                'z': float(grasp_pose.position.z)
            },
            'quality_score': float(grasp_pose.quality_score),
            'grasp_width': float(grasp_pose.width),
            'approach_direction': grasp_pose.approach_direction
        }
        
        self.data['grasp_detections'].append(record)
        
        # Limit data size
        if len(self.data['grasp_detections']) > 1000:
            self.data['grasp_detections'] = self.data['grasp_detections'][-1000:]
        
        self.data['metadata']['total_calls'] += 1
    
    def sync_from_run_history(self):
        """Sync CLIP, GraspNet, OBB, and Pixel-to-Real data from history files
        so the /api/data endpoint stays populated even when service nodes are not running."""
        try:
            from pathlib import Path
            package_path = Path(__file__).parent.parent

            timestamp = datetime.now().isoformat()

            # Rebuild lists from all stored runs (most recent first for display)
            new_clip   = []
            new_grasp  = []
            new_pixel  = []
            new_obb    = []

            history_file = package_path / 'vision_runs_history.json'
            runs = []
            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        runs = data
                except Exception:
                    runs = []

            for run in runs:
                run_ts    = run.get('meta', {}).get('timestamp', timestamp)
                run_no    = run.get('meta', {}).get('run_no', 0)
                objects   = run.get('objects', [])

                for obj in objects:
                    # ── CLIP ──────────────────────────────────────────────────
                    clip_conf = obj.get('clip_confidence', '')
                    label     = obj.get('label', '')
                    if clip_conf != '' and label:
                        new_clip.append({
                            'timestamp':    run_ts,
                            'test_id':      f"{run_no}:{obj.get('object_id', '')}",
                            'bbox': {
                                'x1': obj.get('bbox_x1', 0),
                                'y1': obj.get('bbox_y1', 0),
                                'x2': obj.get('bbox_x2', 0),
                                'y2': obj.get('bbox_y2', 0),
                            },
                            'label':         label,
                            'confidence':    float(clip_conf),
                            'top1_accuracy': float(clip_conf) >= 0.5,
                        })

                    # ── GraspNet ──────────────────────────────────────────────
                    grasp = obj.get('grasp', {})
                    if obj.get('has_grasp') and grasp:
                        pix = grasp.get('pixel', {})
                        wld = grasp.get('world', {})
                        new_grasp.append({
                            'timestamp':    run_ts,
                            'test_id':      f"{run_no}:{obj.get('object_id', '')}",
                            'object_id':    obj.get('object_id', ''),
                            'pixel_position': {'u': pix.get('u', 0), 'v': pix.get('v', 0)},
                            'world_position': {
                                'x': wld.get('x', 0.0),
                                'y': wld.get('y', 0.0),
                                'z': wld.get('z', 0.0),
                            },
                            'quality_score':     float(grasp.get('quality_score', 0.0)),
                            'grasp_width':       float(grasp.get('grasp_width', 0.0)),
                            'approach_direction': grasp.get('approach_direction', ''),
                            'bbox': [obj.get('bbox_x1', 0), obj.get('bbox_y1', 0),
                                     obj.get('bbox_x2', 0), obj.get('bbox_y2', 0)],
                        })

                    # ── Pixel-to-Real ─────────────────────────────────────────
                    world = obj.get('world', {})
                    if world and world.get('x') is not None:
                        new_pixel.append({
                            'timestamp': run_ts,
                            'test_id':   f"{run_no}:{obj.get('object_id', '')}",
                            'input':     {'u': world.get('u', 0), 'v': world.get('v', 0)},
                            'output':    {'x': float(world.get('x', 0.0)),
                                          'y': float(world.get('y', 0.0)),
                                          'z': float(world.get('z', 0.0))},
                        })

                    # ── OBB ───────────────────────────────────────────────────
                    obb_angle = obj.get('obb_angle_deg', '')
                    if obb_angle != '':
                        new_obb.append({
                            'timestamp':   run_ts,
                            'test_id':     f"{run_no}:{obj.get('object_id', '')}",
                            'object_id':   obj.get('object_id', ''),
                            'label':       obj.get('label', ''),
                            'angle_deg':   float(obb_angle),
                            'theta_rad':   float(obj.get('obb_theta_rad', 0.0)),
                            'width_px':    float(obj.get('obb_width_px', 0.0)),
                            'height_px':   float(obj.get('obb_height_px', 0.0)),
                            'center_u':    float(obj.get('obb_center_u', 0.0)),
                            'center_v':    float(obj.get('obb_center_v', 0.0)),
                            'sam_confidence': float(obj.get('sam_confidence', 0.0)),
                            'bbox': [obj.get('bbox_x1', 0), obj.get('bbox_y1', 0),
                                     obj.get('bbox_x2', 0), obj.get('bbox_y2', 0)],
                        })

            # Also merge records from /vision/classify_bbox_filtered direct calls
            filtered_file = Path(__file__).parent.parent / 'classify_filtered_history.json'
            if filtered_file.exists():
                try:
                    with open(filtered_file, 'r') as f:
                        filtered_records = json.load(f)
                    if filtered_records:
                        new_clip = (new_clip + filtered_records)[-1000:]
                except Exception:
                    pass

            # Only update if we got new data (avoids overwriting live topic data with empty)
            if new_clip:
                self.data['clip_classifications'] = new_clip[-1000:]
            if new_grasp:
                self.data['grasp_detections'] = new_grasp[-1000:]
            if new_pixel:
                self.data['pixel_to_real'] = new_pixel[-1000:]
            if new_obb:
                self.data['obb_detections'] = new_obb[-1000:]

        except Exception as e:
            self.get_logger().warn(f'sync_from_run_history failed: {e}')

    def publish_data(self):
        """Publish benchmark data to topic"""
        msg = String()
        msg.data = json.dumps(self.data, cls=_ROSJSONEncoder)
        self.data_publisher.publish(msg)

    def clear_data_callback(self, request, response):
        """Clear all benchmark data (in-memory + persistent JSON files)"""
        self.data = {
            'pixel_to_real': [],
            'sam_detections': [],
            'clip_classifications': [],
            'grasp_detections': [],
            'obb_detections': [],
            'scene_understanding': [],
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'total_calls': 0
            }
        }

        # Also wipe the persistent history files so sync_from_run_history
        # doesn't immediately repopulate from stale data
        package_path = Path(__file__).parent.parent
        files_to_clear = [
            'vision_runs_history.json',
            'classify_filtered_history.json',
            'classify_all_history.json',
            'obb_bb_history.json',
        ]
        for fname in files_to_clear:
            fpath = package_path / fname
            if fpath.exists():
                try:
                    with open(fpath, 'w') as f:
                        json.dump([], f)
                except Exception as e:
                    self.get_logger().warn(f'Could not clear {fname}: {e}')

        response.success = True
        response.message = "Benchmark data cleared"
        self.get_logger().info('Benchmark data cleared (memory + files)')
        return response
    
    def start_http_server(self):
        """Start HTTP server for dashboard"""
        # Resolve dashboard HTML using ament share directory (correct for installed packages)
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory('vision')
            html_dir = Path(share_dir) / 'dashboard'
        except Exception:
            html_dir = Path(__file__).parent.parent / 'dashboard'

        # History JSON files written next to the installed module
        package_path = Path(__file__).parent.parent

        self.get_logger().info(f'Dashboard HTML dir: {html_dir}')

        # Fallback: create default HTML if share dir has no index.html
        html_file = html_dir / 'index.html'
        if not html_file.exists():
            self.get_logger().warn(f'Dashboard HTML not found at {html_file}')
            self.get_logger().warn('Creating basic HTML file...')
            html_dir.mkdir(exist_ok=True)
            self.create_default_html(html_file)

        node_logger = self.get_logger()

        # Custom handler that serves files and provides API endpoints
        class DashboardHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, dashboard_node=None, **kwargs):
                self.dashboard_node = dashboard_node
                super().__init__(*args, directory=str(html_dir), **kwargs)

            # ── helpers ────────────────────────────────────────────────────
            def _json_response(self, data, status=200):
                body = json.dumps(data, cls=_ROSJSONEncoder).encode()
                self.send_response(status)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(body)

            def _read_json_file(self, path, default):
                try:
                    if path.exists():
                        with open(path, 'r') as f:
                            return json.load(f)
                except Exception:
                    pass
                return default

            def _write_json_file(self, path, data):
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)

            # ── OPTIONS (CORS pre-flight) ───────────────────────────────────
            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()

            # ── GET ────────────────────────────────────────────────────────
            def do_GET(self):
                if self.path == '/api/data':
                    self._json_response(self.dashboard_node.data)

                elif self.path == '/api/run-history':
                    history_file = package_path / 'vision_runs_history.json'
                    data = self._read_json_file(history_file, [])
                    self._json_response(data)

                elif self.path == '/api/find-object-history':
                    fo_file = package_path / 'find_object_history.json'
                    data = self._read_json_file(fo_file, [])
                    self._json_response(data)

                elif self.path == '/api/classify-all-history':
                    ca_file = package_path / 'classify_all_history.json'
                    data = self._read_json_file(ca_file, [])
                    self._json_response(data)

                elif self.path == '/api/classify-filtered-history':
                    cf_file = package_path / 'classify_filtered_history.json'
                    data = self._read_json_file(cf_file, [])
                    self._json_response(data)

                elif self.path == '/api/obb-bb-history':
                    obb_file = package_path / 'obb_bb_history.json'
                    data = self._read_json_file(obb_file, [])
                    self._json_response(data)

                else:
                    super().do_GET()

            # ── POST ───────────────────────────────────────────────────────
            def do_POST(self):
                length = int(self.headers.get('Content-Length', 0))
                body   = self.rfile.read(length) if length else b'{}'
                try:
                    payload = json.loads(body)
                except Exception:
                    payload = {}

                if self.path == '/api/find-object':
                    self._handle_find_object(payload)

                elif self.path == '/api/find-object-verdict':
                    self._handle_verdict(payload)

                elif self.path == '/api/find-object-clear':
                    fo_file = package_path / 'find_object_history.json'
                    self._write_json_file(fo_file, [])
                    self._json_response({'ok': True})

                elif self.path == '/api/clip-verdict':
                    self._handle_clip_verdict(payload)

                else:
                    self._json_response({'error': 'unknown endpoint'}, 404)

            # ── /api/clip-verdict ─────────────────────────────────────────
            def _handle_clip_verdict(self, payload):
                """Set human-in-the-loop top1_accuracy verdict for a CLIP record."""
                test_id = payload.get('test_id')
                verdict = payload.get('verdict')  # True / False
                if test_id is None or verdict is None:
                    self._json_response({'error': 'test_id and verdict required'}, 400)
                    return

                cf_file = package_path / 'classify_filtered_history.json'
                history = self._read_json_file(cf_file, [])
                updated = False
                for entry in history:
                    if entry.get('test_id') == test_id:
                        entry['top1_accuracy'] = bool(verdict)
                        updated = True
                        break
                if updated:
                    self._write_json_file(cf_file, history)
                    self._json_response({'ok': True})
                else:
                    self._json_response({'error': f'test_id {test_id} not found'}, 404)

            # ── /api/find-object ──────────────────────────────────────────
            def _handle_find_object(self, payload):
                """Call /find_object ROS2 service and persist result."""
                import subprocess, shlex
                label = payload.get('label', '').strip()
                if not label:
                    self._json_response({'error': 'label is required'}, 400)
                    return

                fo_file = package_path / 'find_object_history.json'
                history = self._read_json_file(fo_file, [])
                call_id = len(history) + 1
                timestamp = datetime.now().isoformat()

                # Call the ROS2 service via subprocess
                cmd = (
                    f"ros2 service call /find_object "
                    f"custom_interfaces/srv/FindObjectReal "
                    f"\"{{label: '{label}'}}\""
                )
                try:
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, timeout=30
                    )
                    output = result.stdout + result.stderr
                    node_logger.info(f'find_object [{label}] stdout: {output[:300]}')

                    # Parse the ROS2 CLI response format
                    entry = _parse_find_object_response(output, label, call_id, timestamp)
                except subprocess.TimeoutExpired:
                    entry = {
                        'call_id': call_id, 'timestamp': timestamp,
                        'label_searched': label, 'success': False,
                        'message': 'Service call timed out (30 s)',
                        'object_id': '', 'bbox': [], 'confidence': 0.0,
                        'x': 0.0, 'y': 0.0, 'z': 0.0, 'theta': 0.0,
                        'verdict': None,
                    }
                except Exception as e:
                    entry = {
                        'call_id': call_id, 'timestamp': timestamp,
                        'label_searched': label, 'success': False,
                        'message': f'Error: {e}',
                        'object_id': '', 'bbox': [], 'confidence': 0.0,
                        'x': 0.0, 'y': 0.0, 'z': 0.0, 'theta': 0.0,
                        'verdict': None,
                    }

                history.append(entry)
                history = history[-50:]
                self._write_json_file(fo_file, history)
                self._json_response(entry)

            # ── /api/find-object-verdict ──────────────────────────────────
            def _handle_verdict(self, payload):
                call_id = payload.get('call_id')
                verdict = payload.get('verdict')  # true / false
                if call_id is None or verdict is None:
                    self._json_response({'error': 'call_id and verdict required'}, 400)
                    return

                fo_file = package_path / 'find_object_history.json'
                history = self._read_json_file(fo_file, [])
                updated = False
                for entry in history:
                    if entry.get('call_id') == call_id:
                        entry['verdict'] = bool(verdict)
                        updated = True
                        break
                if updated:
                    self._write_json_file(fo_file, history)
                    self._json_response({'ok': True})
                else:
                    self._json_response({'error': f'call_id {call_id} not found'}, 404)

            def log_message(self, fmt, *args):
                pass  # suppress HTTP access log noise

        def _parse_find_object_response(output, label, call_id, timestamp):
            """Parse ros2 service call CLI output into a dict."""
            import re
            entry = {
                'call_id': call_id, 'timestamp': timestamp,
                'label_searched': label, 'success': False,
                'message': output.strip()[:500],
                'object_id': '', 'bbox': [], 'confidence': 0.0,
                'x': 0.0, 'y': 0.0, 'z': 0.0, 'theta': 0.0,
                'verdict': None,
            }
            try:
                # success field
                m = re.search(r'success=(\w+)', output)
                if m:
                    entry['success'] = m.group(1).lower() == 'true'
                # message field
                m = re.search(r"message='([^']*)'", output)
                if m:
                    entry['message'] = m.group(1)
                # object_id
                m = re.search(r"object_id='([^']*)'", output)
                if m:
                    entry['object_id'] = m.group(1)
                # bbox
                m = re.search(r'bbox=\[([^\]]*)\]', output)
                if m:
                    try:
                        entry['bbox'] = [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
                    except Exception:
                        pass
                # confidence
                m = re.search(r'confidence=([\d.]+)', output)
                if m:
                    entry['confidence'] = float(m.group(1))
                # x, y, z, theta
                for field in ('x', 'y', 'z', 'theta'):
                    m = re.search(rf'{field}=([-\d.]+)', output)
                    if m:
                        entry[field] = float(m.group(1))
            except Exception as e:
                node_logger.warn(f'Response parse error: {e}')
            return entry

        # Create handler with dashboard_node reference
        def handler_with_node(*args, **kwargs):
            return DashboardHandler(*args, dashboard_node=self, **kwargs)

        # Start server in separate thread
        server = HTTPServer(('0.0.0.0', 8080), handler_with_node)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        self.get_logger().info(f'HTTP server started on http://localhost:8080')
        self.get_logger().info(f'Find Object page: http://localhost:8080/find_object.html')
        self.get_logger().info(f'Serving files from: {html_dir}')
    
    def create_default_html(self, html_file):
        """Create a basic HTML file if none exists"""
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Vision Benchmark Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        h1 { color: #333; }
        p { color: #666; }
    </style>
</head>
<body>
    <h1>Vision Benchmark Dashboard</h1>
    <p>Dashboard HTML file will be created. Please restart the node.</p>
</body>
</html>"""
        html_file.write_text(html_content)


def main(args=None):
    rclpy.init(args=args)
    
    dashboard = BenchmarkDashboard()
    
    try:
        rclpy.spin(dashboard)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
