#!/usr/bin/env python3
"""
Refactored GraspNet Detector - Simplified and Clean

Provides grasp pose detection using geometric methods.
Integrates with SAM detector and pixel-to-real conversion.

Services:
    /vision/detect_grasp - Detect grasps for all objects from SAM
    /vision/detect_grasp_bb - Detect grasp in specific bounding box
    /vision/run_pipeline - Auto-detect when SAM publishes

Publisher:
    /vision/grasp_poses - Grasp pose results
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Custom interfaces
try:
    from custom_interfaces.srv import DetectObjects, DetectGraspBBox, PixelToReal
    from custom_interfaces.msg import SAMDetections, GraspPose
    CUSTOM_INTERFACES = True
except ImportError:
    CUSTOM_INTERFACES = False
    print("Custom interfaces not available. Limited functionality.")

# GraspNet imports (optional)
try:
    # Import GraspNet if available
    GRASPNET_AVAILABLE = False  # Set to True if you have GraspNet installed
except ImportError:
    GRASPNET_AVAILABLE = False

from vision_refactor.utils.common import VisionNodeBase, OpenCVWindow, draw_bbox


class GraspNetDetector(VisionNodeBase):
    """
    Simplified grasp pose detector
    
    Detects grasp poses using geometric analysis of object contours.
    Converts pixel coordinates to world coordinates via pixel_to_real service.
    """
    
    def __init__(self):
        super().__init__('graspnet_detector')
        
        # Grasp detection state
        self.latest_grasps: List[Dict] = []
        self.latest_detections: List[Dict] = []
        
        # Output directory
        self.output_dir = Path.home() / "graspnet_outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        # Visualization
        self.window = OpenCVWindow("GraspNet Detector", 900, 700)
        
        # Setup ROS components
        self.setup_camera_subscriptions()
        self.setup_services()
        self.setup_publishers()
        self.setup_subscribers()
        
        # Service clients
        self.setup_service_clients()
        
        # Visualization timer
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)
        
        self.get_logger().info("GraspNet Detector initialized")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info("Services: /vision/detect_grasp, /vision/detect_grasp_bb")
    
    def setup_services(self):
        """Create grasp detection services"""
        self.grasp_service = self.create_service(
            Trigger,
            '/vision/detect_grasp',
            self.detect_grasp_callback,
            callback_group=self.callback_group
        )
        
        if CUSTOM_INTERFACES:
            self.grasp_bb_service = self.create_service(
                DetectGraspBBox,
                '/vision/detect_grasp_bb',
                self.detect_grasp_bb_callback,
                callback_group=self.callback_group
            )
        
        self.pipeline_service = self.create_service(
            Trigger,
            '/vision/run_pipeline',
            self.run_pipeline_callback,
            callback_group=self.callback_group
        )
    
    def setup_publishers(self):
        """Create grasp pose publishers"""
        self.grasp_pub = self.create_publisher(
            PoseStamped,
            '/vision/grasp_poses',
            self.service_qos
        )
    
    def setup_subscribers(self):
        """Subscribe to SAM detections"""
        if CUSTOM_INTERFACES:
            self.sam_sub = self.create_subscription(
                SAMDetections,
                '/vision/sam_detections',
                self.sam_detections_callback,
                self.service_qos
            )
    
    def setup_service_clients(self):
        """Create service clients for other pipeline components"""
        if CUSTOM_INTERFACES:
            self.detect_objects_client = self.create_client(
                DetectObjects,
                '/vision/detect_objects'
            )
            
            self.pixel_to_real_client = self.create_client(
                PixelToReal,
                '/pixel_to_real'
            )
    
    def detect_grasp_callback(self, request, response):
        """Detect grasps for all objects from SAM detector"""
        try:
            if not self.has_camera_data():
                response.success = False
                response.message = "No camera data available"
                return response
            
            # Get detections from SAM
            if CUSTOM_INTERFACES and self.detect_objects_client.service_is_ready():
                detect_request = DetectObjects.Request()
                detect_response = self.detect_objects_client.call(detect_request)
                
                if not detect_response.success:
                    response.success = False
                    response.message = "Failed to get object detections"
                    return response
                
                # Convert response to detection list
                detections = self.parse_detection_response(detect_response)
            else:
                # Use latest detections if service unavailable
                detections = self.latest_detections
            
            if not detections:
                response.success = False
                response.message = "No objects detected"
                return response
            
            # Detect grasps for all objects
            all_grasps = []
            for detection in detections:
                bbox = detection.get('bbox')
                if bbox:
                    grasps = self.detect_grasps_in_bbox(
                        self.latest_rgb, 
                        self.latest_depth, 
                        bbox
                    )
                    all_grasps.extend(grasps)
            
            self.latest_grasps = all_grasps
            
            # Publish grasp poses
            self.publish_grasp_poses(all_grasps)
            
            response.success = True
            response.message = f"Detected {len(all_grasps)} grasp poses for {len(detections)} objects"
            
            self.get_logger().info(f"Grasp detection: {len(all_grasps)} poses for {len(detections)} objects")
            
        except Exception as e:
            response.success = False
            response.message = f"Grasp detection failed: {str(e)}"
            self.get_logger().error(f"Grasp detection error: {e}")
        
        return response
    
    def detect_grasp_bb_callback(self, request, response):
        """Detect grasp in specific bounding box"""
        try:
            if not self.has_camera_data():
                response.success = False
                response.message = "No camera data available"
                return response
            
            bbox = [request.x1, request.y1, request.x2, request.y2]
            grasps = self.detect_grasps_in_bbox(
                self.latest_rgb, 
                self.latest_depth, 
                bbox
            )
            
            if not grasps:
                response.success = False
                response.message = "No grasps detected in region"
                return response
            
            # Return best grasp
            best_grasp = max(grasps, key=lambda g: g.get('quality', 0))
            
            response.success = True
            response.message = f"Found grasp with quality {best_grasp['quality']:.3f}"
            response.grasp_x = best_grasp['world_position'][0]
            response.grasp_y = best_grasp['world_position'][1]
            response.grasp_z = best_grasp['world_position'][2]
            response.quality_score = best_grasp['quality']
            response.grasp_width = best_grasp['width']
            response.approach_angle = best_grasp['angle']
            
        except Exception as e:
            response.success = False
            response.message = f"Grasp detection failed: {str(e)}"
        
        return response
    
    def run_pipeline_callback(self, request, response):
        """Wait for SAM detections and process automatically"""
        try:
            response.success = True
            response.message = "Pipeline mode enabled - waiting for SAM detections"
            self.get_logger().info("Pipeline mode: waiting for SAM detections")
        except Exception as e:
            response.success = False
            response.message = f"Pipeline setup failed: {str(e)}"
        
        return response
    
    def sam_detections_callback(self, msg: SAMDetections):
        """Automatically detect grasps when SAM publishes detections"""
        try:
            if not self.has_camera_data():
                return
            
            self.latest_detections = []
            all_grasps = []
            
            # Process each detection
            for detection in msg.detections:
                # Store detection info
                det_info = {
                    'object_id': detection.object_id,
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'center': detection.center,
                    'distance': detection.distance_cm
                }
                self.latest_detections.append(det_info)
                
                # Detect grasps for this object
                grasps = self.detect_grasps_in_bbox(
                    self.latest_rgb,
                    self.latest_depth,
                    detection.bbox
                )
                
                # Add object info to grasps
                for grasp in grasps:
                    grasp['object_id'] = detection.object_id
                    grasp['detection_confidence'] = detection.confidence
                
                all_grasps.extend(grasps)
            
            self.latest_grasps = all_grasps
            
            # Publish grasp poses
            if all_grasps:
                self.publish_grasp_poses(all_grasps)
            
            self.get_logger().info(f"Auto-grasp: {len(all_grasps)} poses for {len(msg.detections)} objects")
            
        except Exception as e:
            self.get_logger().error(f"Auto-grasp detection error: {e}")
    
    def detect_grasps_in_bbox(self, rgb_image: np.ndarray, depth_image: Optional[np.ndarray], 
                             bbox: List[int]) -> List[Dict]:
        """
        Detect grasp poses within bounding box using geometric analysis
        
        Args:
            rgb_image: RGB image
            depth_image: Depth image (optional)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            List of grasp pose dictionaries
        """
        x1, y1, x2, y2 = bbox
        h, w = rgb_image.shape[:2]
        
        # Clamp to image bounds
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return []
        
        # Extract region
        roi_rgb = rgb_image[y1:y2, x1:x2]
        roi_h, roi_w = roi_rgb.shape[:2]
        
        if roi_h < 10 or roi_w < 10:
            return []
        
        grasps = []
        
        # Convert to grayscale for contour analysis
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 50:  # Skip tiny contours
                continue
            
            # Get contour properties
            try:
                # Fit ellipse for orientation
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    
                    # Convert center back to full image coordinates
                    center_x = int(center[0] + x1)
                    center_y = int(center[1] + y1)
                    
                    # Calculate grasp quality based on shape
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        quality = min(0.95, max(0.1, circularity))
                    else:
                        quality = 0.5
                    
                    # Calculate grasp width (smaller axis of ellipse)
                    width = min(axes) * 0.01  # Convert pixels to meters (approximate)
                    width = max(0.02, min(0.08, width))  # Clamp to reasonable gripper range
                    
                    # Convert pixel to world coordinates
                    world_pos = self.convert_pixel_to_world(center_x, center_y)
                    
                    if world_pos:
                        grasp = {
                            'pixel_position': [center_x, center_y],
                            'world_position': world_pos,
                            'quality': quality,
                            'width': width,
                            'angle': float(angle),
                            'area': area,
                            'axes': axes
                        }
                        grasps.append(grasp)
                        
            except Exception:
                continue
        
        # Sort by quality
        grasps.sort(key=lambda g: g['quality'], reverse=True)
        
        # Return top 3 grasps
        return grasps[:3]
    
    def convert_pixel_to_world(self, u: int, v: int) -> Optional[Tuple[float, float, float]]:
        """Convert pixel coordinates to world coordinates"""
        if not CUSTOM_INTERFACES or not self.pixel_to_real_client.service_is_ready():
            # Fallback: simple depth-based conversion
            return self.simple_pixel_to_world(u, v)
        
        try:
            request = PixelToReal.Request()
            request.u = u
            request.v = v
            
            response = self.pixel_to_real_client.call(request)
            
            if response.success:
                return (response.x, response.y, response.z)
            else:
                return self.simple_pixel_to_world(u, v)
                
        except Exception:
            return self.simple_pixel_to_world(u, v)
    
    def simple_pixel_to_world(self, u: int, v: int) -> Tuple[float, float, float]:
        """Simple fallback pixel to world conversion"""
        if self.latest_depth is not None and self.camera_info is not None:
            try:
                h, w = self.latest_depth.shape
                u = max(0, min(u, w - 1))
                v = max(0, min(v, h - 1))
                
                depth = self.latest_depth[v, u]
                if depth > 0:
                    # Simple pinhole camera model
                    fx = self.camera_info.k[0] if self.camera_info.k[0] > 0 else 500.0
                    fy = self.camera_info.k[4] if self.camera_info.k[4] > 0 else 500.0
                    cx = self.camera_info.k[2] if self.camera_info.k[2] > 0 else w / 2
                    cy = self.camera_info.k[5] if self.camera_info.k[5] > 0 else h / 2
                    
                    if self.real_hardware:
                        depth_m = depth / 1000.0  # mm to m
                    else:
                        depth_m = depth  # already in meters
                    
                    x = (u - cx) * depth_m / fx
                    y = (v - cy) * depth_m / fy
                    z = depth_m
                    
                    return (x, y, z)
            except Exception:
                pass
        
        # Default fallback position
        return (0.5, 0.0, 0.8)
    
    def parse_detection_response(self, response) -> List[Dict]:
        """Parse DetectObjects response into detection list"""
        detections = []
        
        try:
            for i in range(response.total_detections):
                bbox_start = i * 4
                bbox = response.bboxes[bbox_start:bbox_start + 4]
                
                detection = {
                    'object_id': response.object_ids[i],
                    'bbox': bbox,
                    'confidence': response.confidences[i],
                    'distance': response.distances[i]
                }
                detections.append(detection)
        except Exception as e:
            self.get_logger().error(f"Failed to parse detections: {e}")
        
        return detections
    
    def publish_grasp_poses(self, grasps: List[Dict]):
        """Publish grasp poses as PoseStamped messages"""
        for grasp in grasps:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_link"
            
            # Position
            world_pos = grasp['world_position']
            pose_msg.pose.position = Point(x=world_pos[0], y=world_pos[1], z=world_pos[2])
            
            # Orientation (simplified)
            angle_rad = np.radians(grasp['angle'])
            pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=np.sin(angle_rad/2), w=np.cos(angle_rad/2))
            
            self.grasp_pub.publish(pose_msg)
    
    def visualization_callback(self):
        """Display grasp detection results"""
        if not self.has_camera_data():
            return
        
        vis_image = self.latest_rgb.copy()
        
        # Draw detected objects
        for detection in self.latest_detections:
            bbox = detection.get('bbox')
            if bbox:
                color = (255, 0, 0)  # Blue for objects
                vis_image = draw_bbox(
                    vis_image, bbox, 
                    label=detection.get('object_id', 'obj'),
                    confidence=detection.get('confidence', 0),
                    color=color
                )
        
        # Draw grasp points
        for grasp in self.latest_grasps:
            pixel_pos = grasp.get('pixel_position')
            if pixel_pos:
                x, y = pixel_pos
                
                # Draw grasp point
                quality = grasp.get('quality', 0)
                color = (0, 255, 0) if quality > 0.5 else (0, 165, 255)  # Green/Orange
                
                cv2.circle(vis_image, (x, y), 8, color, -1)
                cv2.circle(vis_image, (x, y), 12, color, 2)
                
                # Draw grasp orientation
                angle = grasp.get('angle', 0)
                length = 30
                end_x = int(x + length * np.cos(np.radians(angle)))
                end_y = int(y + length * np.sin(np.radians(angle)))
                cv2.line(vis_image, (x, y), (end_x, end_y), color, 2)
                
                # Draw quality text
                quality_text = f"{quality:.2f}"
                cv2.putText(vis_image, quality_text, (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add info overlay
        info_text = f"Objects: {len(self.latest_detections)} | Grasps: {len(self.latest_grasps)}"
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
        node = GraspNetDetector()
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
        print(f"Failed to start GraspNet Detector: {e}")
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()