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
    
    def publish_data(self):
        """Publish benchmark data to topic"""
        msg = String()
        msg.data = json.dumps(self.data, cls=_ROSJSONEncoder)
        self.data_publisher.publish(msg)
    
    def clear_data_callback(self, request, response):
        """Clear all benchmark data"""
        self.data = {
            'pixel_to_real': [],
            'sam_detections': [],
            'clip_classifications': [],
            'grasp_detections': [],
            'scene_understanding': [],
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'total_calls': 0
            }
        }
        
        response.success = True
        response.message = "Benchmark data cleared"
        
        self.get_logger().info('Benchmark data cleared')
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

        # History JSON is written by simple_sam_detector next to the installed module
        package_path = Path(__file__).parent.parent

        self.get_logger().info(f'Dashboard HTML dir: {html_dir}')

        # Fallback: create default HTML if share dir has no index.html
        html_file = html_dir / 'index.html'
        if not html_file.exists():
            self.get_logger().warn(f'Dashboard HTML not found at {html_file}')
            self.get_logger().warn('Creating basic HTML file...')
            html_dir.mkdir(exist_ok=True)
            self.create_default_html(html_file)
        
        # Custom handler that serves files from html_dir and provides data endpoint
        class DashboardHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, dashboard_node=None, **kwargs):
                self.dashboard_node = dashboard_node
                super().__init__(*args, directory=str(html_dir), **kwargs)
            
            def do_GET(self):
                if self.path == '/api/data':
                    # Serve benchmark data as JSON
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    data_json = json.dumps(self.dashboard_node.data, cls=_ROSJSONEncoder)
                    self.wfile.write(data_json.encode())
                elif self.path == '/api/run-history':
                    # Serve vision_runs_history.json from workspace root
                    history_file = package_path / 'vision_runs_history.json'
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    if history_file.exists():
                        self.wfile.write(history_file.read_bytes())
                    else:
                        self.wfile.write(b'[]')
                else:
                    # Serve static files
                    super().do_GET()
        
        # Create handler with dashboard_node reference
        def handler_with_node(*args, **kwargs):
            return DashboardHandler(*args, dashboard_node=self, **kwargs)
        
        # Start server in separate thread
        server = HTTPServer(('0.0.0.0', 8080), handler_with_node)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        
        self.get_logger().info(f'HTTP server started on http://localhost:8080')
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
