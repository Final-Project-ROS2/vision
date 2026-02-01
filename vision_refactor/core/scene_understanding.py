#!/usr/bin/env python3
"""
Refactored Scene Understanding - Clean Spatial Analysis

Analyzes spatial relationships between detected objects.
Integrates with SAM detector and CLIP classifier.

Services:
    /vision/understand_scene - Analyze spatial relationships
    /vision/run_pipeline - Auto-analyze when SAM publishes

Publisher:
    /vision/scene_understanding - Scene analysis results
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
import cv2
import numpy as np
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import Counter
from pathlib import Path

# Custom interfaces
try:
    from custom_interfaces.srv import DetectObjects, UnderstandScene
    from custom_interfaces.msg import (
        SAMDetections, SceneUnderstanding, 
        SceneObject, SceneRelation
    )
    CUSTOM_INTERFACES = True
except ImportError:
    CUSTOM_INTERFACES = False
    print("Custom interfaces not available. Limited functionality.")

from vision_refactor.utils.common import VisionNodeBase, OpenCVWindow, draw_bbox


# Spatial relationship types
SPATIAL_RELATIONS = {
    "left_of": "is to the left of",
    "right_of": "is to the right of", 
    "above": "is above",
    "below": "is below",
    "near": "is near",
    "far_from": "is far from",
    "aligned_horizontal": "is horizontally aligned with",
    "aligned_vertical": "is vertically aligned with",
}


class SceneUnderstanding(VisionNodeBase):
    """
    Simplified scene understanding for spatial relationship analysis
    
    Analyzes spatial relationships between objects detected by SAM
    and classified by CLIP to build a scene understanding model.
    """
    
    def __init__(self):
        super().__init__('scene_understanding')
        
        # Scene analysis state
        self.latest_scene: Optional[Dict] = None
        self.latest_objects: List[Dict] = []
        self.latest_relations: List[Dict] = []
        
        # Output directory
        self.output_dir = Path.home() / "scene_understanding_outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        # Visualization
        self.window = OpenCVWindow("Scene Understanding", 1200, 900)
        
        # Setup ROS components
        self.setup_camera_subscriptions()
        self.setup_services()
        self.setup_publishers()
        self.setup_subscribers()
        self.setup_service_clients()
        
        # Visualization timer
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)
        
        self.get_logger().info("Scene Understanding initialized")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info("Services: /vision/understand_scene, /vision/run_pipeline")
    
    def setup_services(self):
        """Create scene understanding services"""
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
    
    def setup_publishers(self):
        """Create scene understanding publishers"""
        if CUSTOM_INTERFACES:
            self.scene_pub = self.create_publisher(
                SceneUnderstanding,
                '/vision/scene_understanding',
                self.service_qos
            )
    
    def setup_subscribers(self):
        """Subscribe to SAM detections for automatic analysis"""
        if CUSTOM_INTERFACES:
            self.sam_sub = self.create_subscription(
                SAMDetections,
                '/vision/sam_detections',
                self.sam_detections_callback,
                self.service_qos
            )
    
    def setup_service_clients(self):
        """Create service clients"""
        if CUSTOM_INTERFACES:
            self.detect_objects_client = self.create_client(
                DetectObjects,
                '/vision/detect_objects'
            )
    
    def understand_scene_callback(self, request, response):
        """Analyze scene from current object detections"""
        try:
            start_time = time.time()
            
            # Get object detections
            if CUSTOM_INTERFACES and self.detect_objects_client.service_is_ready():
                detect_request = DetectObjects.Request()
                detect_response = self.detect_objects_client.call(detect_request)
                
                if not detect_response.success:
                    response.success = False
                    response.message = "Failed to get object detections"
                    return response
                
                objects = self.parse_detection_response(detect_response)
            else:
                objects = self.latest_objects
            
            if not objects:
                response.success = False
                response.message = "No objects detected for scene analysis"
                return response
            
            # Analyze spatial relationships
            relations = self.analyze_spatial_relationships(objects)
            
            # Build scene understanding
            scene_data = self.build_scene_understanding(objects, relations)
            
            # Store results
            self.latest_scene = scene_data
            self.latest_objects = objects
            self.latest_relations = relations
            
            # Publish scene understanding
            if CUSTOM_INTERFACES:
                self.publish_scene_understanding(scene_data)
            
            processing_time = time.time() - start_time
            
            response.success = True
            response.message = json.dumps({
                'total_objects': len(objects),
                'total_relations': len(relations),
                'scene_description': scene_data.get('description', ''),
                'processing_time_s': processing_time
            })
            
            self.get_logger().info(f"Scene analysis: {len(objects)} objects, {len(relations)} relations ({processing_time:.3f}s)")
            
        except Exception as e:
            response.success = False
            response.message = f"Scene analysis failed: {str(e)}"
            self.get_logger().error(f"Scene analysis error: {e}")
        
        return response
    
    def run_pipeline_callback(self, request, response):
        """Enable automatic scene analysis when SAM publishes"""
        try:
            response.success = True
            response.message = "Scene understanding pipeline enabled - waiting for SAM detections"
            self.get_logger().info("Scene pipeline: waiting for SAM detections")
        except Exception as e:
            response.success = False
            response.message = f"Pipeline setup failed: {str(e)}"
        
        return response
    
    def sam_detections_callback(self, msg: SAMDetections):
        """Automatically analyze scene when SAM publishes detections"""
        try:
            if not self.has_camera_data():
                return
            
            # Convert SAM detections to object list
            objects = []
            for detection in msg.detections:
                obj = {
                    'object_id': detection.object_id,
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'center': detection.center,
                    'distance_cm': detection.distance_cm,
                    'area': detection.area,
                    'class_label': 'unknown'  # Will be filled by CLIP if available
                }
                objects.append(obj)
            
            # Analyze spatial relationships
            relations = self.analyze_spatial_relationships(objects)
            
            # Build scene understanding
            scene_data = self.build_scene_understanding(objects, relations)
            
            # Store results
            self.latest_scene = scene_data
            self.latest_objects = objects
            self.latest_relations = relations
            
            # Publish scene understanding
            if CUSTOM_INTERFACES:
                self.publish_scene_understanding(scene_data)
            
            self.get_logger().info(f"Auto-scene analysis: {len(objects)} objects, {len(relations)} relations")
            
        except Exception as e:
            self.get_logger().error(f"Auto-scene analysis error: {e}")
    
    def analyze_spatial_relationships(self, objects: List[Dict]) -> List[Dict]:
        """
        Analyze spatial relationships between objects
        
        Args:
            objects: List of object dictionaries with bbox and center info
            
        Returns:
            List of spatial relationship dictionaries
        """
        relations = []
        
        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i >= j:  # Avoid self-comparison and duplicates
                    continue
                
                # Calculate spatial relations between obj_a and obj_b
                obj_relations = self.compute_spatial_relations(obj_a, obj_b)
                relations.extend(obj_relations)
        
        return relations
    
    def compute_spatial_relations(self, obj_a: Dict, obj_b: Dict) -> List[Dict]:
        """
        Compute spatial relationships between two objects
        
        Args:
            obj_a: First object dictionary
            obj_b: Second object dictionary
            
        Returns:
            List of relation dictionaries
        """
        relations = []
        
        # Extract positions
        bbox_a = obj_a['bbox']
        bbox_b = obj_b['bbox']
        center_a = obj_a['center']
        center_b = obj_b['center']
        
        x1_a, y1_a, x2_a, y2_a = bbox_a
        x1_b, y1_b, x2_b, y2_b = bbox_b
        cx_a, cy_a = center_a
        cx_b, cy_b = center_b
        
        # Calculate 2D distance
        distance_2d = np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
        
        # Calculate 3D distance if depth available
        distance_3d = -1.0
        if 'distance_cm' in obj_a and 'distance_cm' in obj_b:
            dist_a = obj_a['distance_cm']
            dist_b = obj_b['distance_cm']
            if dist_a > 0 and dist_b > 0:
                distance_3d = abs(dist_a - dist_b)
        
        # Thresholds
        near_threshold = 100.0  # pixels
        alignment_threshold = 30.0  # pixels
        
        # Determine relationships
        
        # Left/Right
        if cx_a < cx_b - alignment_threshold:
            relations.append(self.create_relation(
                obj_a, obj_b, "left_of", 0.8, distance_2d, distance_3d
            ))
        elif cx_a > cx_b + alignment_threshold:
            relations.append(self.create_relation(
                obj_a, obj_b, "right_of", 0.8, distance_2d, distance_3d
            ))
        
        # Above/Below  
        if cy_a < cy_b - alignment_threshold:
            relations.append(self.create_relation(
                obj_a, obj_b, "above", 0.8, distance_2d, distance_3d
            ))
        elif cy_a > cy_b + alignment_threshold:
            relations.append(self.create_relation(
                obj_a, obj_b, "below", 0.8, distance_2d, distance_3d
            ))
        
        # Near/Far
        if distance_2d < near_threshold:
            relations.append(self.create_relation(
                obj_a, obj_b, "near", 0.7, distance_2d, distance_3d
            ))
        else:
            relations.append(self.create_relation(
                obj_a, obj_b, "far_from", 0.6, distance_2d, distance_3d
            ))
        
        # Alignment
        if abs(cy_a - cy_b) < alignment_threshold:
            relations.append(self.create_relation(
                obj_a, obj_b, "aligned_horizontal", 0.9, distance_2d, distance_3d
            ))
        
        if abs(cx_a - cx_b) < alignment_threshold:
            relations.append(self.create_relation(
                obj_a, obj_b, "aligned_vertical", 0.9, distance_2d, distance_3d
            ))
        
        return relations
    
    def create_relation(self, obj_a: Dict, obj_b: Dict, relation_type: str,
                       confidence: float, distance_2d: float, distance_3d: float) -> Dict:
        """Create a spatial relation dictionary"""
        return {
            'subject_id': obj_a['object_id'],
            'subject_label': obj_a.get('class_label', 'unknown'),
            'relation': relation_type,
            'object_id': obj_b['object_id'], 
            'object_label': obj_b.get('class_label', 'unknown'),
            'confidence': confidence,
            'distance_2d': distance_2d,
            'distance_3d': distance_3d,
            'description': f"{obj_a['object_id']} {SPATIAL_RELATIONS.get(relation_type, relation_type)} {obj_b['object_id']}"
        }
    
    def build_scene_understanding(self, objects: List[Dict], relations: List[Dict]) -> Dict:
        """
        Build comprehensive scene understanding data structure
        
        Args:
            objects: List of object dictionaries
            relations: List of spatial relations
            
        Returns:
            Scene understanding dictionary
        """
        # Count object types
        labels = [obj.get('class_label', 'unknown') for obj in objects]
        label_counts = Counter(labels)
        
        # Generate scene description
        description = self.generate_scene_description(objects, relations)
        
        # Calculate scene statistics
        total_distance = sum([obj.get('distance_cm', 0) for obj in objects if obj.get('distance_cm', 0) > 0])
        valid_distances = [obj.get('distance_cm', 0) for obj in objects if obj.get('distance_cm', 0) > 0]
        avg_distance = np.mean(valid_distances) if valid_distances else 0.0
        
        scene_data = {
            'scene_id': f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'total_objects': len(objects),
            'objects': objects,
            'total_relations': len(relations),
            'relations': relations,
            'object_labels': list(label_counts.keys()),
            'object_counts': list(label_counts.values()),
            'description': description,
            'statistics': {
                'average_distance_cm': avg_distance,
                'scene_density': len(objects),  # objects per scene
                'relation_density': len(relations) / max(1, len(objects))  # relations per object
            }
        }
        
        return scene_data
    
    def generate_scene_description(self, objects: List[Dict], relations: List[Dict]) -> str:
        """Generate natural language scene description"""
        if not objects:
            return "Empty scene with no detected objects."
        
        num_objects = len(objects)
        desc_parts = []
        
        # Basic object count
        desc_parts.append(f"Scene contains {num_objects} object{'s' if num_objects != 1 else ''}.")
        
        # Describe key relationships
        if relations:
            # Count relation types
            relation_types = [rel['relation'] for rel in relations]
            relation_counts = Counter(relation_types)
            
            # Describe most common relations
            for rel_type, count in relation_counts.most_common(3):
                human_rel = SPATIAL_RELATIONS.get(rel_type, rel_type.replace('_', ' '))
                desc_parts.append(f"{count} object pair{'s' if count != 1 else ''} show {human_rel} relationship.")
        
        return " ".join(desc_parts)
    
    def parse_detection_response(self, response) -> List[Dict]:
        """Parse DetectObjects response into object list"""
        objects = []
        
        try:
            for i in range(response.total_detections):
                bbox_start = i * 4
                bbox = response.bboxes[bbox_start:bbox_start + 4]
                
                # Calculate center
                x1, y1, x2, y2 = bbox
                center = [(x1 + x2) // 2, (y1 + y2) // 2]
                
                obj = {
                    'object_id': response.object_ids[i],
                    'bbox': bbox,
                    'confidence': response.confidences[i],
                    'distance_cm': response.distances[i],
                    'center': center,
                    'area': (x2 - x1) * (y2 - y1),
                    'class_label': 'unknown'
                }
                objects.append(obj)
        except Exception as e:
            self.get_logger().error(f"Failed to parse detections: {e}")
        
        return objects
    
    def publish_scene_understanding(self, scene_data: Dict):
        """Publish scene understanding as ROS message"""
        if not CUSTOM_INTERFACES:
            return
        
        try:
            msg = SceneUnderstanding()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            msg.scene_id = scene_data['scene_id']
            msg.total_objects = scene_data['total_objects']
            msg.scene_description = scene_data['description']
            
            # Add object information
            for obj in scene_data['objects']:
                scene_obj = SceneObject()
                scene_obj.object_id = obj['object_id']
                scene_obj.class_label = obj.get('class_label', 'unknown')
                scene_obj.bbox = obj['bbox']
                scene_obj.center = obj['center']
                scene_obj.confidence = obj['confidence']
                scene_obj.distance_cm = obj.get('distance_cm', 0.0)
                scene_obj.area = obj.get('area', 0)
                scene_obj.has_grasp = False  # Could be populated by GraspNet
                
                msg.objects.append(scene_obj)
            
            # Add relations
            for rel in scene_data['relations']:
                scene_rel = SceneRelation()
                scene_rel.subject_id = rel['subject_id']
                scene_rel.subject_label = rel['subject_label']
                scene_rel.relation = rel['relation']
                scene_rel.object_id = rel['object_id']
                scene_rel.object_label = rel['object_label']
                scene_rel.confidence = rel['confidence']
                scene_rel.distance_2d = rel['distance_2d']
                scene_rel.distance_3d = rel['distance_3d']
                
                msg.all_relations.append(scene_rel)
            
            # Add statistics
            stats = scene_data['statistics']
            msg.average_distance_cm = stats['average_distance_cm']
            msg.scene_density = stats['scene_density']
            
            self.scene_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish scene understanding: {e}")
    
    def visualization_callback(self):
        """Display scene understanding results"""
        if not self.has_camera_data():
            return
        
        vis_image = self.latest_rgb.copy()
        
        # Draw objects with spatial relationship indicators
        for obj in self.latest_objects:
            bbox = obj.get('bbox')
            if bbox:
                color = (0, 255, 0)  # Green for objects
                label = f"{obj.get('object_id', 'obj')} ({obj.get('class_label', 'unknown')})"
                
                vis_image = draw_bbox(
                    vis_image, bbox,
                    label=label,
                    confidence=obj.get('confidence', 0),
                    color=color
                )
        
        # Draw relationship lines for key spatial relations
        relation_colors = {
            'left_of': (255, 0, 0),      # Blue
            'right_of': (255, 0, 0),     # Blue  
            'above': (0, 255, 255),      # Yellow
            'below': (0, 255, 255),      # Yellow
            'near': (0, 255, 0),         # Green
            'aligned_horizontal': (255, 0, 255),  # Magenta
            'aligned_vertical': (255, 255, 0)     # Cyan
        }
        
        # Draw lines between related objects
        for relation in self.latest_relations:
            if relation['confidence'] > 0.7:  # Only high-confidence relations
                subject_id = relation['subject_id']
                object_id = relation['object_id']
                rel_type = relation['relation']
                
                # Find object centers
                subject_center = None
                object_center = None
                
                for obj in self.latest_objects:
                    if obj['object_id'] == subject_id:
                        subject_center = obj['center']
                    elif obj['object_id'] == object_id:
                        object_center = obj['center']
                
                if subject_center and object_center:
                    color = relation_colors.get(rel_type, (128, 128, 128))
                    cv2.line(vis_image, tuple(subject_center), tuple(object_center), color, 2)
        
        # Add scene information overlay
        if self.latest_scene:
            scene = self.latest_scene
            info_lines = [
                f"Objects: {scene['total_objects']}",
                f"Relations: {scene['total_relations']}",
                f"Avg Distance: {scene['statistics']['average_distance_cm']:.1f}cm"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 25
                cv2.putText(vis_image, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add description (truncated if too long)
            desc = scene['description']
            if len(desc) > 80:
                desc = desc[:77] + "..."
            
            cv2.putText(vis_image, desc, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self.window.show(vis_image)
    
    def destroy_node(self):
        """Clean shutdown"""
        self.window.close()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = SceneUnderstanding()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            node.destroy_node()
    
    except Exception as e:
        print(f"Failed to start Scene Understanding: {e}")
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()