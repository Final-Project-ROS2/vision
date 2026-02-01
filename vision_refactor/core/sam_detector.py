#!/usr/bin/env python3
"""
Refactored SAM Detector - Simplified and Clean

Provides object detection using OpenCV-based segmentation methods.
Focuses on core functionality without unnecessary complexity.

Services:
    /vision/run_pipeline - Trigger detection and publish results
    /vision/detect_objects - Get detection results directly

Publisher:
    /vision/sam_detections - Detection results for other pipeline components
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from std_msgs.msg import String
import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Optional

# Import custom interfaces if available
try:
    from custom_interfaces.msg import SAMDetections, SAMDetection
    from custom_interfaces.srv import DetectObjects
    CUSTOM_INTERFACES = True
except ImportError:
    CUSTOM_INTERFACES = False
    print("Custom interfaces not available. Limited functionality.")

from vision_refactor.utils.common import VisionNodeBase, OpenCVWindow, draw_bbox


class SAMDetector(VisionNodeBase):
    """
    Simplified SAM-style object detector using OpenCV
    
    Detects objects through contour analysis and morphological operations.
    Publishes results for use by CLIP, GraspNet, and Scene Understanding.
    """
    
    def __init__(self):
        super().__init__('sam_detector')
        
        # Detection state
        self.latest_detections: List[Dict] = []
        self.frame_counter = 0
        
        # Visualization
        self.window = OpenCVWindow("SAM Detector", 800, 600)
        
        # Setup camera subscriptions
        self.setup_camera_subscriptions()
        
        # Create services
        self.setup_services()
        
        # Create publishers
        self.setup_publishers()
        
        # Visualization timer
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)
        
        self.get_logger().info("SAM Detector initialized")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info("Services: /vision/run_pipeline, /vision/detect_objects")
        self.get_logger().info("Publisher: /vision/sam_detections")
    
    def setup_services(self):
        """Create detection services"""
        self.pipeline_service = self.create_service(
            Trigger,
            '/vision/run_pipeline',
            self.run_pipeline_callback,
            callback_group=self.callback_group
        )
        
        if CUSTOM_INTERFACES:
            self.detect_objects_service = self.create_service(
                DetectObjects,
                '/vision/detect_objects',
                self.detect_objects_callback,
                callback_group=self.callback_group
            )
    
    def setup_publishers(self):
        """Create detection publishers"""
        if CUSTOM_INTERFACES:
            self.detection_pub = self.create_publisher(
                SAMDetections,
                '/vision/sam_detections',
                self.service_qos
            )
        
        # Status publisher for debugging
        self.status_pub = self.create_publisher(
            String,
            '/vision/status',
            10
        )
    
    def run_pipeline_callback(self, request, response):
        """Run detection and publish results"""
        try:
            if not self.has_camera_data():
                response.success = False
                response.message = "No camera data available"
                return response
            
            # Detect objects
            start_time = time.time()
            detections = self.detect_objects(self.latest_rgb)
            detection_time = time.time() - start_time
            
            self.latest_detections = detections
            self.frame_counter += 1
            
            # Publish results
            if CUSTOM_INTERFACES and detections:
                self.publish_sam_detections(detections)
            
            # Update status
            status_msg = String()
            status_msg.data = f"SAM: {len(detections)} objects detected in {detection_time:.3f}s"
            self.status_pub.publish(status_msg)
            
            response.success = True
            response.message = f"Detected {len(detections)} objects"
            
            self.get_logger().info(f"Detection complete: {len(detections)} objects in {detection_time:.3f}s")
            
        except Exception as e:
            response.success = False
            response.message = f"Detection failed: {str(e)}"
            self.get_logger().error(f"Detection error: {e}")
        
        return response
    
    def detect_objects_callback(self, request, response):
        """Return detection results directly in service response"""
        try:
            if not self.has_camera_data():
                response.success = False
                response.message = "No camera data available"
                return response
            
            # Detect objects
            detections = self.detect_objects(self.latest_rgb)
            self.latest_detections = detections
            
            # Fill response arrays
            response.success = True
            response.message = f"Detected {len(detections)} objects"
            response.total_detections = len(detections)
            
            # Convert to parallel arrays for ROS service
            for i, det in enumerate(detections):
                response.object_ids.append(f"obj_{i}")
                response.bboxes.extend(det['bbox'])  # [x1,y1,x2,y2] flattened
                response.confidences.append(det['confidence'])
                response.distances.append(det.get('distance', 100.0))  # Default distance
            
        except Exception as e:
            response.success = False
            response.message = f"Detection failed: {str(e)}"
            self.get_logger().error(f"Detection error: {e}")
        
        return response
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Core object detection using OpenCV
        
        Args:
            image: Input BGR image
            
        Returns:
            List of detection dictionaries
        """
        if image is None:
            return []
        
        h, w = image.shape[:2]
        detections = []
        
        # Convert to grayscale and preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        min_area = (w * h) * 0.001  # 0.1% of image
        max_area = (w * h) * 0.8    # 80% of image
        min_size = 20
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < min_area or area > max_area:
                continue
            
            # Get bounding box
            x, y, box_w, box_h = cv2.boundingRect(contour)
            
            if box_w < min_size or box_h < min_size:
                continue
            
            # Calculate confidence based on contour properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            confidence = min(0.95, max(0.1, circularity))
            
            # Create detection
            detection = {
                'bbox': [x, y, x + box_w, y + box_h],
                'confidence': confidence,
                'area': area,
                'contour': contour.tolist(),
                'center': [x + box_w // 2, y + box_h // 2],
                'size': [box_w, box_h]
            }
            
            # Estimate distance if depth available
            if self.latest_depth is not None:
                detection['distance'] = self.estimate_distance(detection['center'])
            else:
                detection['distance'] = 100.0  # Default distance in cm
            
            detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def estimate_distance(self, center: List[int]) -> float:
        """Estimate distance to object center using depth data"""
        if self.latest_depth is None:
            return 100.0
        
        try:
            cx, cy = center
            h, w = self.latest_depth.shape
            
            # Clamp to image bounds
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))
            
            # Get depth value (in meters for simulation, mm for real)
            depth_value = self.latest_depth[cy, cx]
            
            if self.real_hardware:
                # Real hardware depth in mm, convert to cm
                return depth_value / 10.0 if depth_value > 0 else 100.0
            else:
                # Simulation depth in meters, convert to cm
                return depth_value * 100.0 if depth_value > 0 else 100.0
                
        except Exception:
            return 100.0
    
    def publish_sam_detections(self, detections: List[Dict]):
        """Publish detections as SAMDetections message"""
        if not CUSTOM_INTERFACES:
            return
        
        msg = SAMDetections()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.frame_id = f"frame_{self.frame_counter:06d}"
        msg.total_detections = len(detections)
        
        for i, det in enumerate(detections):
            sam_det = SAMDetection()
            sam_det.object_id = f"obj_{i}"
            sam_det.bbox = det['bbox']
            sam_det.confidence = det['confidence']
            sam_det.area = int(det['area'])
            sam_det.center = det['center']
            sam_det.distance_cm = det['distance']
            
            msg.detections.append(sam_det)
        
        self.detection_pub.publish(msg)
    
    def visualization_callback(self):
        """Display detection results"""
        if not self.has_camera_data():
            return
        
        vis_image = self.latest_rgb.copy()
        
        # Draw detections
        for i, det in enumerate(self.latest_detections):
            color = (0, 255, 0) if det['confidence'] > 0.5 else (0, 165, 255)  # Green/Orange
            label = f"obj_{i}"
            
            vis_image = draw_bbox(
                vis_image, 
                det['bbox'], 
                label=label,
                confidence=det['confidence'],
                color=color
            )
        
        # Add info overlay
        info_text = f"Objects: {len(self.latest_detections)} | Frame: {self.frame_counter}"
        cv2.putText(vis_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.window.show(vis_image)
    
    def destroy_node(self):
        """Clean shutdown"""
        self.window.close()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = SAMDetector()
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
        print(f"Failed to start SAM Detector: {e}")
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()