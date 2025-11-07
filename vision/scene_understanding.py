#!/usr/bin/env python3
"""
Scene Understanding Node

Services:
    1. /vision/understand_scene
       Analyze spatial relationships from all detected objects
       Calls /vision/detect_objects internally to get bounding boxes + labels
       ros2 service call /vision/understand_scene std_srvs/srv/Trigger
    
    2. /vision/run_pipeline
       Run complete pipeline: SAM + CLIP + GraspNet + Scene Understanding
       Subscribes to /vision/sam_detections
       Automatically analyzes scene when SAM detections are published
       ros2 service call /vision/run_pipeline std_srvs/srv/Trigger

Publishes:
    /vision/scene_understanding (SceneUnderstanding message)

Setup:
    Terminal 1: ros2 run vision simple_sam_detector
    Terminal 2: ros2 run vision clip_classifier
    Terminal 3: ros2 run vision graspnet_detector
    Terminal 4: ros2 run vision scene_understanding
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter
from pathlib import Path

# Import custom interfaces
try:
    from custom_interfaces.srv import DetectObjects, UnderstandScene, DetectGrasps
    from custom_interfaces.msg import (
        SAMDetections, SAMDetection,
        SceneUnderstanding, SceneObject, SceneRelation
    )
    CUSTOM_INTERFACES_AVAILABLE = True
except ImportError:
    CUSTOM_INTERFACES_AVAILABLE = False
    # Fallback type hints
    SAMDetections = None
    DetectObjects = None
    UnderstandScene = None
    DetectGrasps = None
    SceneUnderstanding = None
    SceneObject = None
    SceneRelation = None
    print("Custom interfaces not available. Build custom_interfaces package first.")


# Spatial relation constants
SPATIAL_RELATIONS = {
    "left_of": "is to the left of",
    "right_of": "is to the right of",
    "above": "is above",
    "below": "is below",
    "in_front_of": "is in front of",
    "behind": "is behind",
    "near": "is near",
    "far_from": "is far from",
    "touching": "is touching",
    "aligned_horizontal": "is horizontally aligned with",
    "aligned_vertical": "is vertically aligned with",
}


class SceneUnderstandingNode(Node):
    """
    Scene Understanding Node
    
    Analyzes spatial relationships between detected objects.
    Integrates SAM (detection) + CLIP (classification) + GraspNet (manipulation).
    
    Subscribes to:
        - /vision/sam_detections (SAMDetections for pipeline mode)
    
    Services:
        - /vision/understand_scene (Analyze all objects from detect_objects)
        - /vision/run_pipeline (Auto-analyze when SAM publishes)
    
    Publishes:
        - /vision/scene_understanding (SceneUnderstanding message)
    """
    
    def __init__(self):
        super().__init__('scene_understanding')
        
        # Create callback group for service calls
        self.callback_group = ReentrantCallbackGroup()
        
        # CV Bridge for ROS<->OpenCV conversion
        self.bridge = CvBridge()
        
        # Latest scene understanding result
        self.latest_scene = None
        self.latest_rgb = None
        
        # Output directory for saving visualizations
        self.output_dir = Path.home() / "scene_understanding_outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        # OpenCV window for visualization
        self.window_name = "Scene Understanding - Spatial Relationships"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 900)
        
        # Service clients
        self.detect_objects_client = None
        self.detect_grasps_client = None
        
        if CUSTOM_INTERFACES_AVAILABLE:
            self.detect_objects_client = self.create_client(
                DetectObjects,
                '/vision/detect_objects',
                callback_group=self.callback_group
            )
            
            self.detect_grasps_client = self.create_client(
                DetectGrasps,
                '/vision/detect_grasp',
                callback_group=self.callback_group
            )
        
        # QoS profiles
        self.detection_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to camera RGB for visualization
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.rgb_callback,
            self.image_qos
        )
        
        # Subscribe to SAM detections for pipeline mode
        if CUSTOM_INTERFACES_AVAILABLE:
            self.sam_sub = self.create_subscription(
                SAMDetections,
                '/vision/sam_detections',
                self.sam_detections_callback,
                10
            )
        
        # Create scene understanding services
        # Always use Trigger for /vision/understand_scene to ensure compatibility
        self.understand_service = self.create_service(
            Trigger,
            '/vision/understand_scene',
            self.understand_scene_callback,
            callback_group=self.callback_group
        )
        
        self.pipeline_service = self.create_service(
            Trigger,
            '/vision/run_pipeline',
            self.run_pipeline_callback,
            callback_group=self.callback_group
        )
        
        # Publisher for scene understanding
        if CUSTOM_INTERFACES_AVAILABLE:
            self.scene_pub = self.create_publisher(
                SceneUnderstanding,
                '/vision/scene_understanding',
                self.detection_qos
            )
        
        # Visualization timer
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("Scene Understanding Node Started")
        self.get_logger().info("=" * 80)
        self.get_logger().info("Subscribed to: /camera/image_raw")
        if CUSTOM_INTERFACES_AVAILABLE:
            self.get_logger().info("Subscribed to: /vision/sam_detections")
        self.get_logger().info(f"Output Directory: {self.output_dir}")
        self.get_logger().info(f"OpenCV Window: '{self.window_name}'")
        self.get_logger().info("Service: /vision/understand_scene")
        self.get_logger().info("Service: /vision/run_pipeline")
        if CUSTOM_INTERFACES_AVAILABLE:
            self.get_logger().info("Publishing to: /vision/scene_understanding")
        self.get_logger().info("=" * 80)
        self.get_logger().info("Usage:")
        self.get_logger().info("  ros2 service call /vision/understand_scene std_srvs/srv/Trigger")
        self.get_logger().info("  ros2 service call /vision/run_pipeline std_srvs/srv/Trigger")
        self.get_logger().info("=" * 80)
    
    def rgb_callback(self, msg: Image):
        """Handle RGB image messages for visualization"""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
    
    def sam_detections_callback(self, msg: 'SAMDetections'):
        """Handle incoming SAM detections for pipeline mode"""
        try:
            self.get_logger().info(f"Received {len(msg.detections)} SAM detections, running scene understanding...")
            
            # Trigger scene understanding
            self._analyze_scene_from_detections()
            
        except Exception as e:
            self.get_logger().error(f"Error handling SAM detections: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def understand_scene_callback(self, request, response):
        """Service callback for /vision/understand_scene - uses Trigger service"""
        try:
            self.get_logger().info("=" * 60)
            self.get_logger().info("Scene Understanding Service Called - Analyzing...")
            self.get_logger().info("=" * 60)
            
            if not CUSTOM_INTERFACES_AVAILABLE:
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": "Custom interfaces not available. Build custom_interfaces package first.",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("Custom interfaces not available")
                return response
            
            # Call /vision/detect_objects to get all detections
            if not self.detect_objects_client.wait_for_service(timeout_sec=5.0):
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": "/vision/detect_objects service not available",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("/vision/detect_objects service not available")
                return response
            
            detect_request = DetectObjects.Request()
            future = self.detect_objects_client.call_async(detect_request)
            
            # Wait for future with timeout using executor
            start_time = time.time()
            timeout = 10.0
            while not future.done() and (time.time() - start_time) < timeout:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
            
            if not future.done():
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": "/vision/detect_objects service call timeout",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error("Service call timeout")
                return response
            
            detect_response = future.result()
            
            if not detect_response.success:
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": f"Object detection failed: {detect_response.error_message}",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().error(f"Object detection failed: {detect_response.error_message}")
                return response
            
            # Call /vision/detect_grasp to get grasp information
            grasp_data = {}
            if self.detect_grasps_client is not None:
                self.get_logger().info("Fetching grasp information...")
                if self.detect_grasps_client.wait_for_service(timeout_sec=2.0):
                    grasp_request = DetectGrasps.Request()
                    grasp_future = self.detect_grasps_client.call_async(grasp_request)
                    
                    # Wait for future with timeout
                    start_time = time.time()
                    timeout = 10.0
                    while not grasp_future.done() and (time.time() - start_time) < timeout:
                        time.sleep(0.01)
                    
                    if grasp_future.done():
                        grasp_response = grasp_future.result()
                        if grasp_response.success:
                            # Map grasp poses to object IDs
                            for grasp_pose in grasp_response.grasp_poses:
                                obj_id = grasp_pose.object_id
                                if obj_id not in grasp_data:
                                    grasp_data[obj_id] = []
                                grasp_data[obj_id].append({
                                    'quality': grasp_pose.quality_score,
                                    'width': grasp_pose.width
                                })
            
            # Build scene understanding from detections
            scene = self._build_scene_understanding(detect_response, grasp_data)
            
            # Store and publish result
            self.latest_scene = scene
            self.scene_pub.publish(scene)
            
            # Visualize scene
            if self.latest_rgb is not None:
                self._visualize_scene(self.latest_rgb, scene)
            
            # Create success response
            response.success = True
            response.message = json.dumps({
                "success": True,
                "scene_id": scene.scene_id,
                "total_objects": scene.total_objects,
                "total_relations": len(scene.all_relations),
                "graspable_objects": scene.graspable_objects,
                "scene_description": scene.scene_description,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"✓ Scene Analysis Complete!")
            self.get_logger().info(f"  Objects: {scene.total_objects}")
            self.get_logger().info(f"  Relations: {len(scene.all_relations)}")
            self.get_logger().info(f"  Graspable: {scene.graspable_objects}")
            self.get_logger().info(f"  Description: {scene.scene_description}")
            self.get_logger().info("=" * 60)
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"Scene understanding error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response
    
    def run_pipeline_callback(self, request, response):
        """Service callback for /vision/run_pipeline - waits for SAM detections"""
        try:
            self.get_logger().info("Pipeline mode activated - waiting for SAM detections on /vision/sam_detections")
            
            response.success = True
            response.message = json.dumps({
                "success": True,
                "message": "Pipeline mode active. Scene understanding will process SAM detections automatically.",
                "listening_to": "/vision/sam_detections",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"Pipeline activation error: {e}")
        
        return response
    
    def _analyze_scene_from_detections(self):
        """Analyze scene from current detections (triggered by SAM callback)"""
        try:
            if not CUSTOM_INTERFACES_AVAILABLE or self.detect_objects_client is None:
                return
            
            # Call detect_objects service
            if not self.detect_objects_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn("detect_objects service not available for scene analysis")
                return
            
            detect_request = DetectObjects.Request()
            future = self.detect_objects_client.call_async(detect_request)
            
            # Wait for future with timeout
            start_time = time.time()
            timeout = 10.0
            while not future.done() and (time.time() - start_time) < timeout:
                time.sleep(0.01)
            
            if not future.done() or not future.result().success:
                self.get_logger().warn("Failed to get object detections for scene analysis")
                return
            
            detect_response = future.result()
            
            # Get grasp data
            grasp_data = {}
            if self.detect_grasps_client is not None:
                if self.detect_grasps_client.wait_for_service(timeout_sec=2.0):
                    grasp_request = DetectGrasps.Request()
                    grasp_future = self.detect_grasps_client.call_async(grasp_request)
                    
                    # Wait for future with timeout
                    start_time = time.time()
                    timeout = 10.0
                    while not grasp_future.done() and (time.time() - start_time) < timeout:
                        time.sleep(0.01)
                    
                    if grasp_future.done():
                        grasp_response = grasp_future.result()
                        if grasp_response.success:
                            for grasp_pose in grasp_response.grasp_poses:
                                obj_id = grasp_pose.object_id
                                if obj_id not in grasp_data:
                                    grasp_data[obj_id] = []
                                grasp_data[obj_id].append({
                                    'quality': grasp_pose.quality_score,
                                    'width': grasp_pose.width
                                })
            
            # Build and publish scene understanding
            scene = self._build_scene_understanding(detect_response, grasp_data)
            self.latest_scene = scene
            self.scene_pub.publish(scene)
            
            # Visualize scene
            if self.latest_rgb is not None:
                self._visualize_scene(self.latest_rgb, scene)
            
            self.get_logger().info(f"Pipeline scene analysis complete: {scene.total_objects} objects")
            
        except Exception as e:
            self.get_logger().error(f"Error in pipeline scene analysis: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def _build_scene_understanding(self, detect_response, grasp_data: Dict) -> 'SceneUnderstanding':
        """
        Build SceneUnderstanding message from detection results
        
        Args:
            detect_response: Response from DetectObjects service
            grasp_data: Dictionary mapping object_id to grasp information
            
        Returns:
            SceneUnderstanding message
        """
        scene = SceneUnderstanding()
        scene.header.stamp = self.get_clock().now().to_msg()
        scene.header.frame_id = "camera_link"
        scene.scene_id = f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract object information
        objects_list = []
        total_distance = 0.0
        distance_count = 0
        
        for i in range(detect_response.total_detections):
            obj = SceneObject()
            obj.object_id = detect_response.object_ids[i]
            obj.class_label = detect_response.object_ids[i]  # Will be updated if labels available
            obj.classification_conf = detect_response.confidences[i]
            obj.bbox = [
                detect_response.bbox_x1[i],
                detect_response.bbox_y1[i],
                detect_response.bbox_x2[i],
                detect_response.bbox_y2[i]
            ]
            
            # Calculate center
            cx = (detect_response.bbox_x1[i] + detect_response.bbox_x2[i]) // 2
            cy = (detect_response.bbox_y1[i] + detect_response.bbox_y2[i]) // 2
            obj.center = [cx, cy]
            
            # Distance
            obj.distance_cm = detect_response.distances_cm[i]
            if obj.distance_cm > 0:
                total_distance += obj.distance_cm
                distance_count += 1
            
            # Grasp information
            if obj.object_id in grasp_data and len(grasp_data[obj.object_id]) > 0:
                obj.has_grasp = True
                obj.grasp_quality = max([g['quality'] for g in grasp_data[obj.object_id]])
            else:
                obj.has_grasp = False
                obj.grasp_quality = 0.0
            
            objects_list.append(obj)
        
        # Compute spatial relationships between all pairs of objects
        all_relations = []
        for i, obj_a in enumerate(objects_list):
            for j, obj_b in enumerate(objects_list):
                if i >= j:  # Skip self and duplicate pairs
                    continue
                
                relations = self._compute_spatial_relations(obj_a, obj_b)
                
                # Add relations to object A
                for rel in relations:
                    obj_a.relations.append(rel)
                    all_relations.append(rel)
        
        scene.objects = objects_list
        scene.total_objects = len(objects_list)
        scene.all_relations = all_relations
        
        # Scene summary
        scene.scene_description = self._generate_scene_description(objects_list, all_relations)
        
        # Count unique object labels
        label_counter = Counter([obj.class_label for obj in objects_list])
        scene.object_labels = list(label_counter.keys())
        scene.object_counts = [label_counter[label] for label in scene.object_labels]
        
        # Count graspable objects
        scene.graspable_objects = sum([1 for obj in objects_list if obj.has_grasp])
        
        # Statistics
        if distance_count > 0:
            scene.average_distance_cm = total_distance / distance_count
        else:
            scene.average_distance_cm = 0.0
        
        # Scene density (objects per square meter) - simplified
        scene.scene_density = float(len(objects_list))  # Placeholder
        
        return scene
    
    def _compute_spatial_relations(self, obj_a: 'SceneObject', obj_b: 'SceneObject') -> List['SceneRelation']:
        """
        Compute spatial relationships between two objects
        
        Args:
            obj_a: First object
            obj_b: Second object
            
        Returns:
            List of SceneRelation messages
        """
        relations = []
        
        # Extract bounding box coordinates
        x1_a, y1_a, x2_a, y2_a = obj_a.bbox
        x1_b, y1_b, x2_b, y2_b = obj_b.bbox
        
        cx_a, cy_a = obj_a.center
        cx_b, cy_b = obj_b.center
        
        # Calculate 2D pixel distance between centers
        distance_2d = np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
        
        # Calculate 3D distance (if depth available)
        distance_3d = -1.0
        if obj_a.distance_cm > 0 and obj_b.distance_cm > 0:
            # Simplified 3D distance using camera distance
            distance_3d = abs(obj_a.distance_cm - obj_b.distance_cm) / 100.0  # Convert to meters
        
        # Thresholds
        near_threshold = 100.0  # pixels
        alignment_threshold = 30.0  # pixels
        
        # Determine spatial relations
        
        # Left/Right
        if cx_a < cx_b - alignment_threshold:
            rel = self._create_relation(obj_a, obj_b, "left_of", 0.9, distance_2d, distance_3d)
            relations.append(rel)
        elif cx_a > cx_b + alignment_threshold:
            rel = self._create_relation(obj_a, obj_b, "right_of", 0.9, distance_2d, distance_3d)
            relations.append(rel)
        
        # Above/Below
        if cy_a < cy_b - alignment_threshold:
            rel = self._create_relation(obj_a, obj_b, "above", 0.9, distance_2d, distance_3d)
            relations.append(rel)
        elif cy_a > cy_b + alignment_threshold:
            rel = self._create_relation(obj_a, obj_b, "below", 0.9, distance_2d, distance_3d)
            relations.append(rel)
        
        # In front / Behind (based on depth)
        if distance_3d > 0:
            if obj_a.distance_cm < obj_b.distance_cm - 5.0:  # 5cm threshold
                rel = self._create_relation(obj_a, obj_b, "in_front_of", 0.85, distance_2d, distance_3d)
                relations.append(rel)
            elif obj_a.distance_cm > obj_b.distance_cm + 5.0:
                rel = self._create_relation(obj_a, obj_b, "behind", 0.85, distance_2d, distance_3d)
                relations.append(rel)
        
        # Near/Far
        if distance_2d < near_threshold:
            rel = self._create_relation(obj_a, obj_b, "near", 0.8, distance_2d, distance_3d)
            relations.append(rel)
        else:
            rel = self._create_relation(obj_a, obj_b, "far_from", 0.7, distance_2d, distance_3d)
            relations.append(rel)
        
        # Touching (bounding boxes overlap)
        if self._boxes_overlap(obj_a.bbox, obj_b.bbox):
            rel = self._create_relation(obj_a, obj_b, "touching", 0.95, distance_2d, distance_3d)
            relations.append(rel)
        
        # Aligned horizontal (similar y-coordinates)
        if abs(cy_a - cy_b) < alignment_threshold:
            rel = self._create_relation(obj_a, obj_b, "aligned_horizontal", 0.85, distance_2d, distance_3d)
            relations.append(rel)
        
        # Aligned vertical (similar x-coordinates)
        if abs(cx_a - cx_b) < alignment_threshold:
            rel = self._create_relation(obj_a, obj_b, "aligned_vertical", 0.85, distance_2d, distance_3d)
            relations.append(rel)
        
        return relations
    
    def _create_relation(self, obj_a: 'SceneObject', obj_b: 'SceneObject', 
                        relation_type: str, confidence: float,
                        distance_2d: float, distance_3d: float) -> 'SceneRelation':
        """Create a SceneRelation message"""
        rel = SceneRelation()
        rel.subject_id = obj_a.object_id
        rel.subject_label = obj_a.class_label
        rel.relation = relation_type
        rel.object_id = obj_b.object_id
        rel.object_label = obj_b.class_label
        rel.confidence = confidence
        rel.distance_2d = distance_2d
        rel.distance_3d = distance_3d
        return rel
    
    def _boxes_overlap(self, bbox_a: List[int], bbox_b: List[int]) -> bool:
        """Check if two bounding boxes overlap"""
        x1_a, y1_a, x2_a, y2_a = bbox_a
        x1_b, y1_b, x2_b, y2_b = bbox_b
        
        # Check if boxes do NOT overlap, then negate
        no_overlap = (x2_a < x1_b or x2_b < x1_a or y2_a < y1_b or y2_b < y1_a)
        return not no_overlap
    
    def _generate_scene_description(self, objects: List['SceneObject'], 
                                   relations: List['SceneRelation']) -> str:
        """
        Generate natural language scene description
        
        Args:
            objects: List of SceneObject
            relations: List of SceneRelation
            
        Returns:
            Natural language description string
        """
        if not objects:
            return "Empty scene with no detected objects."
        
        # Count objects
        num_objects = len(objects)
        num_graspable = sum([1 for obj in objects if obj.has_grasp])
        
        # Build description
        desc_parts = []
        desc_parts.append(f"Scene contains {num_objects} object{'s' if num_objects != 1 else ''}.")
        
        if num_graspable > 0:
            desc_parts.append(f"{num_graspable} object{'s are' if num_graspable != 1 else ' is'} graspable.")
        
        # Describe some key relationships
        if len(relations) > 0:
            # Find most confident relations
            sorted_rels = sorted(relations, key=lambda r: r.confidence, reverse=True)
            top_relations = sorted_rels[:min(5, len(sorted_rels))]
            
            rel_descriptions = []
            for rel in top_relations:
                if rel.relation in SPATIAL_RELATIONS:
                    rel_text = f"{rel.subject_id} {SPATIAL_RELATIONS[rel.relation]} {rel.object_id}"
                    rel_descriptions.append(rel_text)
            
            if rel_descriptions:
                desc_parts.append("Key spatial relationships: " + "; ".join(rel_descriptions) + ".")
        
        return " ".join(desc_parts)
    
    def _visualize_scene(self, rgb_image: np.ndarray, scene: 'SceneUnderstanding'):
        """
        Visualize scene understanding with spatial relationships
        
        Args:
            rgb_image: RGB image to draw on
            scene: SceneUnderstanding message
        """
        vis_image = rgb_image.copy()
        
        # Define colors for different relation types
        relation_colors = {
            "left_of": (255, 100, 100),      # Light Blue
            "right_of": (100, 100, 255),     # Light Red
            "above": (255, 255, 100),        # Light Cyan
            "below": (100, 255, 255),        # Light Yellow
            "in_front_of": (100, 255, 100),  # Light Green
            "behind": (255, 100, 255),       # Light Magenta
            "near": (0, 255, 0),             # Green
            "far_from": (128, 128, 128),     # Gray
            "touching": (0, 0, 255),         # Red
            "aligned_horizontal": (255, 165, 0),  # Orange
            "aligned_vertical": (255, 0, 255),    # Magenta
        }
        
        # Draw all bounding boxes and labels
        for obj in scene.objects:
            x1, y1, x2, y2 = obj.bbox
            cx, cy = obj.center
            
            # Choose color based on grasp availability
            if obj.has_grasp:
                bbox_color = (0, 255, 0)  # Green if graspable
                thickness = 3
            else:
                bbox_color = (0, 165, 255)  # Orange if not graspable
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), bbox_color, thickness)
            
            # Draw center point
            cv2.circle(vis_image, (cx, cy), 5, bbox_color, -1)
            cv2.circle(vis_image, (cx, cy), 7, (255, 255, 255), 2)
            
            # Draw label with background
            label = f"{obj.object_id}"
            if obj.has_grasp:
                label += f" G:{obj.grasp_quality:.2f}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, 
                         (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0] + 4, y1), 
                         bbox_color, -1)
            cv2.putText(vis_image, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw distance if available
            if obj.distance_cm > 0:
                dist_text = f"{obj.distance_cm:.1f}cm"
                cv2.putText(vis_image, dist_text, (cx + 10, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, bbox_color, 1)
        
        # Draw relationship lines between objects
        drawn_relations = set()  # Track drawn relations to avoid duplicates
        
        for rel in scene.all_relations:
            # Find the objects
            obj_a = next((o for o in scene.objects if o.object_id == rel.subject_id), None)
            obj_b = next((o for o in scene.objects if o.object_id == rel.object_id), None)
            
            if obj_a is None or obj_b is None:
                continue
            
            # Skip if we already drew a line between these objects
            pair_key = tuple(sorted([rel.subject_id, rel.object_id]))
            if pair_key in drawn_relations:
                continue
            
            # Only draw high-confidence, important relations
            if rel.confidence < 0.7:
                continue
            
            # Skip "far_from" relations to reduce clutter
            if rel.relation == "far_from":
                continue
            
            cx_a, cy_a = obj_a.center
            cx_b, cy_b = obj_b.center
            
            # Get color for this relation type
            line_color = relation_colors.get(rel.relation, (200, 200, 200))
            
            # Draw line between centers
            cv2.line(vis_image, (cx_a, cy_a), (cx_b, cy_b), line_color, 2, cv2.LINE_AA)
            
            # Draw small circles at endpoints
            cv2.circle(vis_image, (cx_a, cy_a), 3, line_color, -1)
            cv2.circle(vis_image, (cx_b, cy_b), 3, line_color, -1)
            
            # Draw relation label at midpoint
            mid_x = (cx_a + cx_b) // 2
            mid_y = (cy_a + cy_b) // 2
            
            rel_text = rel.relation.replace('_', ' ')
            text_size = cv2.getTextSize(rel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            
            # Draw text background
            cv2.rectangle(vis_image,
                         (mid_x - text_size[0]//2 - 2, mid_y - text_size[1] - 2),
                         (mid_x + text_size[0]//2 + 2, mid_y + 2),
                         (0, 0, 0), -1)
            
            cv2.putText(vis_image, rel_text, 
                       (mid_x - text_size[0]//2, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, line_color, 1, cv2.LINE_AA)
            
            drawn_relations.add(pair_key)
        
        # Add scene description at the top
        title = f"Scene Understanding | Objects: {scene.total_objects} | Relations: {len(scene.all_relations)}"
        cv2.putText(vis_image, title, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add graspable count
        grasp_text = f"Graspable: {scene.graspable_objects}/{scene.total_objects}"
        cv2.putText(vis_image, grasp_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add natural language description (split into multiple lines if too long)
        desc = scene.scene_description
        max_width = 80
        if len(desc) > max_width:
            words = desc.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= max_width:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(' '.join(current_line))
        else:
            lines = [desc]
        
        # Draw description lines at bottom
        y_offset = vis_image.shape[0] - 20 - (len(lines) * 25)
        for i, line in enumerate(lines[:3]):  # Max 3 lines
            cv2.putText(vis_image, line, (10, y_offset + (i * 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = self.output_dir / f"scene_understanding_{timestamp}.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        self.get_logger().info(f"   Visualization saved: {vis_path}")
        
        # Update display
        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)
    
    def visualization_callback(self):
        """Display current scene understanding"""
        if self.latest_rgb is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera image...", (100, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow(self.window_name, blank)
            cv2.waitKey(1)
            return
        
        if self.latest_scene is None:
            display_img = self.latest_rgb.copy()
            cv2.putText(display_img, "Call service to analyze scene", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_img, "ros2 service call /vision/understand_scene std_srvs/srv/Trigger", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(display_img, "Can be called multiple times to update", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
            cv2.imshow(self.window_name, display_img)
            cv2.waitKey(1)
            return
        
        # Display latest scene understanding
        self._visualize_scene(self.latest_rgb, self.latest_scene)
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        from rclpy.executors import MultiThreadedExecutor
        node = SceneUnderstandingNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
