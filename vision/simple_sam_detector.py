#!/usr/bin/env python3
"""
Simple SAM Vision Detection Node

Provides object detection and segmentation using OpenCV-based methods.
Subscribes to camera topics and provides detection services.

Services:
    1. /vision/run_pipeline
       Trigger SAM detection and publish to /vision/sam_detections topic (message - many frame)
       CLIP automatically subscribes and classifies regions
       ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
    
    2. /vision/detect_objects
       Trigger SAM detection and return results directly in service response (service - one frame only)
       Returns parallel arrays of object_ids, bboxes, confidences, and distances
       ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects
    
    3. /vision/show_depth_image
       Display depth camera visualization
       ros2 service call /vision/show_depth_image std_srvs/srv/Trigger

Setup:
    Terminal 1: ros2 run vision simple_sam_detector
    Terminal 2: ros2 service call /vision/run_pipeline std_srvs/srv/Trigger
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from custom_interfaces.msg import SAMDetections, SAMDetection
from custom_interfaces.srv import DetectObjects
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from std_msgs.msg import Header
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple


class SimpleSAMDetector(Node):
    """
    Simple SAM-based object detector
    
    Subscribes to:
        - /camera/image_raw (RGB images from Gazebo camera)
        - /camera/depth/image_raw (Depth images for distance estimation)
    
    Publishes:
        - /vision/sam_detections (SAMDetections message with all detections)
    
    Services:
        - /vision/run_pipeline (Trigger detection and publish to topic)
        - /vision/detect_objects (Trigger detection and return results in response)
        - /vision/show_depth_image (Display depth visualization)
    
    Display:
        - Shows live camera feed with detections in OpenCV window
    """
    
    def __init__(self, single_shot_mode=False):
        super().__init__('simple_sam_detector')
        
        # Mode configuration - Default to single shot for faster service response
        self.single_shot_mode = True  # Force single shot mode for service efficiency
        self.continuous_detection = False
        
        # CV Bridge for ROS<->OpenCV conversion
        self.bridge = CvBridge()
        
        # Latest image from camera
        self.latest_rgb = None
        self.captured_frame = None  # Single captured frame for detection
        self.frame_captured = False
        self.latest_depth = None  # For distance estimation if available
        self.latest_detections = []
        self.frame_counter = 0
        
        # QoS profile for image subscription
        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to camera
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.rgb_callback,
            self.image_qos
        )
        
        # Subscribe to depth  (for Graspnet)
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            self.image_qos
        )
        
        # Detection service
        self.detection_service = self.create_service(
            Trigger,
            '/vision/run_pipeline',
            self.run_pipeline_callback
        )
        
        # Direct detection service (returns results in response)
        self.detect_objects_service = self.create_service(
            DetectObjects,
            '/vision/detect_objects',
            self.detect_objects_callback
        )
        
        # Depth display service
        self.depth_display_service = self.create_service(
            Trigger,
            '/vision/show_depth_image',
            self.show_depth_callback
        )
        
        # Publisher for detection results
        self.detection_publisher = self.create_publisher(
            SAMDetections,  # Placeholder - will be SAMDetections after build
            '/vision/sam_detections',
            10
        )
        self.get_logger().info("Publisher: /vision/sam_detections (SAMDetections)")

        # Status/heartbeat publisher to ensure global visibility and easy debugging
        self.status_publisher = self.create_publisher(
            String,
            '/vision/status',
            10
        )

        # Record a start time for status messages
        self._node_start_time = self.get_clock().now()
        
        # OpenCV window setup
        self.window_name = "SAM Object Detection - /camera/image_raw"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        # Timer for continuous visualization (30 Hz)
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)
        
        mode_str = "SERVICE-BASED (OPTIMIZED)"
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Simple SAM Detector Started [{mode_str}]")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Subscribing to: /camera/image_raw")
        self.get_logger().info(f"Subscribing to: /camera/depth/image_raw")
        self.get_logger().info(f"Will capture ONE frame for efficient detection")
        self.get_logger().info(f"Service: /vision/run_pipeline (publish to topic)")
        self.get_logger().info(f"Service: /vision/detect_objects (return in response)")
        self.get_logger().info(f"Service: /vision/show_depth_image")
        self.get_logger().info(f"Publisher: /vision/sam_detections")
        self.get_logger().info(f"OpenCV Window: '{self.window_name}'")
        self.get_logger().info(f"Optimized: Only detects when service is called")
        self.get_logger().info("=" * 80)
        self.get_logger().info("Run pipeline: ros2 service call /vision/run_pipeline std_srvs/srv/Trigger")
        self.get_logger().info("Get results:  ros2 service call /vision/detect_objects custom_interfaces/srv/DetectObjects")
        self.get_logger().info("Show depth:   ros2 service call /vision/show_depth_image std_srvs/srv/Trigger")
        self.get_logger().info("=" * 80)

        # Announce presence shortly after start so topics appear in `ros2 topic list`
        # and keep advertising status periodicallyh so the `/vision/*` namespace is visible.
        self._startup_timer = self.create_timer(0.5, self._startup_announce)
        self._heartbeat_timer = self.create_timer(5.0, self._heartbeat_callback)
    
    def rgb_callback(self, msg: Image):
        """Handle incoming RGB images from /camera/image_raw"""
        try:
            # Convert ROS Image message to OpenCV format (BGR8)
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_counter += 1
            
            if not self.frame_captured:
                self.captured_frame = self.latest_rgb.copy()
                self.frame_captured = True
                self.get_logger().info(f"Captured frame {self.frame_counter} for detection")
            
            # In continuous mode, detect on every frame (disabled by default now)
            if self.continuous_detection:
                self.latest_detections = self._detect_objects(self.latest_rgb)
                
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    # (Timers started in __init__)        
    
    def depth_callback(self, msg: Image):
        """Handle incoming depth images (optional, for distance estimation)"""
        try:
            # Convert ROS Image to OpenCV format (float32 or uint16 depending on encoding)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Clip any invalid depths (NaN or Inf)
            depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            self.latest_depth = depth_image
            
            # Log first successful depth capture
            if not hasattr(self, '_depth_logged'):
                self.get_logger().info(f"Depth image received: shape={depth_image.shape}, dtype={depth_image.dtype}")
                self._depth_logged = True
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
    
    def run_pipeline_callback(self, request, response):
        """Service callback for /vision/run_pipeline - triggers detection and publishes to topic"""
        try:
            # Use captured frame instead of latest_rgb for consistency
            frame_to_use = self.captured_frame if self.frame_captured else self.latest_rgb
            
            if frame_to_use is None:
                response.success = False
                response.message = json.dumps({
                    "success": False,
                    "error": "No image available from /camera/image_raw",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, indent=2)
                self.get_logger().warn("No image received yet")
                return response
            
            self.get_logger().info("=" * 80)
            self.get_logger().info("Running SAM detection on captured frame...")
            self.get_logger().info(f"Frame shape: {frame_to_use.shape}")
            self.get_logger().info("=" * 80)
            
            # Run detection on captured frame
            self.latest_detections = self._detect_objects(frame_to_use)
            
            # Build JSON response in the requested schema
            detection_data = self._build_detection_schema()
            
            # Publish detections as ROS2 message
            self._publish_detections_ros()
            
            response.success = True
            response.message = json.dumps(detection_data, indent=2)
            
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"Detection complete: {len(self.latest_detections)} objects found")
            self.get_logger().info("=" * 80)
            
            # Print JSON output with bounding boxes
            self.get_logger().info("JSON OUTPUT (with bounding boxes):")
            self.get_logger().info("=" * 80)
            self.get_logger().info(response.message)
            self.get_logger().info("=" * 80)
            
            # Print detection details in readable format
            self.get_logger().info("Bounding Boxes Summary:")
            for i, det in enumerate(self.latest_detections):
                bbox = det['bbox']
                distance = det.get('distance_cm', 'N/A')
                self.get_logger().info(
                    f"   [{i}] {det['class_name']}: bbox={bbox}, "
                    f"confidence={det['confidence']:.2f}, distance={distance}"
                )
            self.get_logger().info("=" * 80)
            
            # Verify bounding boxes are in output
            bbox_count = len([d for d in detection_data.get('detections', [{}])[0].get('detections', []) if 'bbox' in d])
            self.get_logger().info(f"Verified: {bbox_count} bounding boxes included in JSON output")
            self.get_logger().info("=" * 80)
            
        except Exception as e:
            response.success = False
            response.message = json.dumps({
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }, indent=2)
            self.get_logger().error(f"Detection error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response

    def detect_objects_callback(self, request, response):
        """Service callback for /vision/detect_objects - returns detection results directly"""
        try:
            # Use captured frame instead of latest_rgb for consistency
            frame_to_use = self.captured_frame if self.frame_captured else self.latest_rgb
            
            if frame_to_use is None:
                response.success = False
                response.total_detections = 0
                response.object_ids = []
                response.bbox_x1 = []
                response.bbox_y1 = []
                response.bbox_x2 = []
                response.bbox_y2 = []
                response.confidences = []
                response.distances_cm = []
                response.error_message = "No image available from /camera/image_raw"
                self.get_logger().warn("No image received yet")
                return response
            
            self.get_logger().info("Running SAM detection and returning results...")
            
            # Run detection on captured frame
            self.latest_detections = self._detect_objects(frame_to_use)
            
            # Build parallel arrays for response
            object_ids = []
            bbox_x1 = []
            bbox_y1 = []
            bbox_x2 = []
            bbox_y2 = []
            confidences = []
            distances_cm = []
            
            for det in self.latest_detections:
                object_ids.append(det['id'])
                bbox = det['bbox']
                bbox_x1.append(bbox[0])
                bbox_y1.append(bbox[1])
                bbox_x2.append(bbox[2])
                bbox_y2.append(bbox[3])
                confidences.append(float(det['confidence']))
                
                # Add distance if available
                distance = det.get('distance_cm')
                distances_cm.append(float(distance) if distance is not None else -1.0)
            
            # Build response
            response.success = True
            response.total_detections = len(self.latest_detections)
            response.object_ids = object_ids
            response.bbox_x1 = bbox_x1
            response.bbox_y1 = bbox_y1
            response.bbox_x2 = bbox_x2
            response.bbox_y2 = bbox_y2
            response.confidences = confidences
            response.distances_cm = distances_cm
            response.error_message = ""
            
            self.get_logger().info(f"Detection complete: {len(self.latest_detections)} objects found")
            
            # Print bounding boxes
            self.get_logger().info("Bounding Boxes:")
            for i in range(len(object_ids)):
                self.get_logger().info(
                    f"  {object_ids[i]}: bbox=[{bbox_x1[i]}, {bbox_y1[i]}, {bbox_x2[i]}, {bbox_y2[i]}], "
                    f"conf={confidences[i]:.2f}, dist={distances_cm[i]:.1f}cm"
                )
            
        except Exception as e:
            response.success = False
            response.total_detections = 0
            response.object_ids = []
            response.bbox_x1 = []
            response.bbox_y1 = []
            response.bbox_x2 = []
            response.bbox_y2 = []
            response.confidences = []
            response.distances_cm = []
            response.error_message = str(e)
            self.get_logger().error(f"Detection error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response

    def _startup_announce(self):
        """One-shot announce to make sure global topics appear after node startup."""
        try:
            # 1) Publish an initial placeholder detections message so the topic is created
            self._publish_detections_ros()

            # 2) Publish a status heartbeat immediately
            ns = self.get_namespace()
            dom = os.environ.get('ROS_DOMAIN_ID', '0')
            uptime_sec = (self.get_clock().now() - self._node_start_time).nanoseconds / 1e9
            status = String()
            status.data = (
                f"simple_sam_detector alive | ns={ns} | domain={dom} | "
                f"uptime={uptime_sec:.1f}s | detections={len(self.latest_detections)}"
            )
            self.status_publisher.publish(status)

            # 3) Log resolved info so users can verify
            self.get_logger().info(
                f"Namespace: '{ns}' | ROS_DOMAIN_ID: {dom} | Publishing '/vision/sam_detections' & '/vision/status'"
            )

            # 4) Optionally list known topics locally (helpful for debugging)
            try:
                topics = dict(self.get_topic_names_and_types())
                visible = [t for t in topics.keys() if t.startswith('/vision')]
                self.get_logger().info(f"Currently visible '/vision*' topics (local graph): {visible}")
            except Exception:
                pass
        finally:
            # Cancel so it runs only once
            try:
                self._startup_timer.cancel()
            except Exception:
                pass

    def _heartbeat_callback(self):
        """Periodic status publisher to keep the '/vision/*' namespace visible on the graph."""
        try:
            ns = self.get_namespace()
            dom = os.environ.get('ROS_DOMAIN_ID', '0')
            uptime_sec = (self.get_clock().now() - self._node_start_time).nanoseconds / 1e9
            status = String()
            status.data = (
                f"simple_sam_detector heartbeat | ns={ns} | domain={dom} | "
                f"uptime={uptime_sec:.1f}s | detections={len(self.latest_detections)}"
            )
            self.status_publisher.publish(status)
        except Exception as e:
            self.get_logger().warn(f"Heartbeat publish failed: {e}")
    
    def show_depth_callback(self, request, response):
        """Service callback for /vision/show_depth_image to display depth visualization"""
        try:
            if self.latest_depth is None:
                response.success = False
                response.message = "No depth image received yet from /camera/depth/image_raw."
                self.get_logger().warn("No depth image available")
                return response
            
            # Normalize depth for visualization
            normalized_depth = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Create window if it doesn't exist
            depth_window = "Depth Camera Image - /camera/depth/image_raw"
            cv2.namedWindow(depth_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(depth_window, 800, 600)
            
            # Add info overlay
            info_text = f"Depth Image | Shape: {self.latest_depth.shape}"
            cv2.putText(
                depth_colormap,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.imshow(depth_window, depth_colormap)
            cv2.waitKey(1)  # Refresh window
            
            response.success = True
            response.message = "Displayed latest depth image in OpenCV window."
            self.get_logger().info(f"Depth image displayed: {self.latest_depth.shape}")
            
        except Exception as e:
            response.success = False
            response.message = f"Error displaying depth image: {str(e)}"
            self.get_logger().error(f"Depth display error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        return response
    
    def _detect_objects(self, rgb_image: np.ndarray) -> List[Dict]:
        """
        Detect objects using OpenCV contour detection (SAM-style segmentation)
        
        Args:
            rgb_image: BGR image from OpenCV
            
        Returns:
            List of detection dictionaries with bbox, mask, confidence
        """
        if rgb_image is None:
            return []
        
        h, w = rgb_image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding for better object separation
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours (external only)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter parameters
        min_area = (w * h) * 0.001  # Minimum 0.1% of image
        max_area = (w * h) * 0.8    # Maximum 80% of image
        min_box_size = 20           # Minimum bounding box dimension
        
        detections = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding box
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Filter small boxes
            if w_box < min_box_size or h_box < min_box_size:
                continue
            
            # Create binary mask for this object
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Calculate confidence based on contour properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            confidence = min(0.95, 0.60 + circularity * 0.35)
            
            # Estimate distance from depth image (if available)
            distance_cm = None
            center_x = x + w_box // 2
            center_y = y + h_box // 2
            if self.latest_depth is not None:
                try:
                    if 0 <= center_y < self.latest_depth.shape[0] and 0 <= center_x < self.latest_depth.shape[1]:
                        depth_value = self.latest_depth[center_y, center_x]
                        # Convert depth to cm (assuming depth is in mm or meters, adjust as needed)
                        if depth_value > 0:
                            distance_cm = float(depth_value) / 10.0  # Adjust conversion factor as needed
                except Exception as e:
                    pass  # Distance estimation failed, leave as None
            
            detection = {
                "id": f"obj_{i}",
                "class_name": "object",  # Generic class, can be enhanced with actual classification
                "confidence": float(confidence),
                "bbox": [x, y, x + w_box, y + h_box],
                "center": [center_x, center_y],
                "area": int(area),
                "distance_cm": distance_cm,
                "mask": mask,
                "contour": contour
            }
            
            detections.append(detection)
        
        return detections
    
    def _build_detection_schema(self) -> Dict:
        """
        Build detection response in the requested JSON schema format
        
        Returns:
            Dictionary matching the schema with detections, summary, and metadata
        """
        # Frame identifier
        frame_id = f"frame_{self.frame_counter:06d}"
        
        # Build detections list
        detections_list = []
        total_distance = 0.0
        distance_count = 0
        
        for det in self.latest_detections:
            detection_obj = {
                "class_name": det.get("class_name", "object"),
                "confidence": round(det["confidence"], 2),
                "bbox": det["bbox"]
            }
            
            # Add distance if available
            if det.get("distance_cm") is not None:
                detection_obj["distance_cm"] = round(det["distance_cm"], 1)
                total_distance += det["distance_cm"]
                distance_count += 1
            
            detections_list.append(detection_obj)
        
        # Calculate average distance
        average_distance = round(total_distance / distance_count, 1) if distance_count > 0 else None
        
        # Build schema
        schema = {
            "success": True,
            "detections": [
                {
                    "image_id": frame_id,
                    "detections": detections_list
                }
            ],
            "summary": {
                "total_detections": len(self.latest_detections),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        # Add average_distance_cm only if we have distance data
        if average_distance is not None:
            schema["summary"]["average_distance_cm"] = average_distance
        
        return schema
    
    def _publish_detections_ros(self):
        """
        Publish detections as ROS2 message for real-time sharing
        
        This enables other nodes (scene_understanding, graspnet, etc.) to 
        subscribe directly without parsing JSON.
        
        Uses SAMDetections message with array of SAMDetection objects.
        """
        try:
            self.get_logger().info(f"Publishing {len(self.latest_detections)} detections to /vision/sam_detections")
            
            # Create SAMDetections message
            msg = SAMDetections()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            msg.image_id = f"frame_{self.frame_counter:06d}"
            
            # Initialize detections array
            msg.detections = []
            
            # Populate detections array
            total_distance = 0.0
            distance_count = 0
            
            for det in self.latest_detections:
                sam_det = SAMDetection()
                sam_det.object_id = det['id']
                sam_det.class_name = det['class_name']
                sam_det.confidence = float(det['confidence'])
                
                # Bbox as [x1, y1, x2, y2] - already in correct format
                sam_det.bbox = det['bbox']
                
                # Center as [x, y]
                sam_det.center = det['center']
                
                # Area as int32
                sam_det.area = int(det['area'])
                
                # Distance in cm (use -1.0 if unavailable)
                distance = det.get('distance_cm')
                sam_det.distance_cm = float(distance) if distance is not None else -1.0
                
                if sam_det.distance_cm > 0:
                    total_distance += sam_det.distance_cm
                    distance_count += 1
                
                # Convert mask to ROS Image message
                sam_det.mask = self.bridge.cv2_to_imgmsg(det['mask'], encoding='mono8')
                
                msg.detections.append(sam_det)
            
            # Summary statistics
            msg.total_detections = len(self.latest_detections)
            
            # Calculate average distance (exclude -1.0 values)
            msg.average_distance_cm = float(total_distance / distance_count) if distance_count > 0 else -1.0
            
            self.detection_publisher.publish(msg)
            self.get_logger().info(f"Published SAMDetections message with {msg.total_detections} detections")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish ROS detections: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    

    def visualization_callback(self):
        """Display camera feed with detections in OpenCV window"""
        # Use captured frame if available, otherwise latest_rgb
        frame_to_display = self.captured_frame if self.frame_captured else self.latest_rgb
        
        if frame_to_display is None:
            # Show waiting message
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank, 
                "Waiting for /camera/image_raw...", 
                (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255, 255, 255), 
                2
            )
            cv2.imshow(self.window_name, blank)
            cv2.waitKey(1)
            return
        
        # Create visualization image
        vis_image = frame_to_display.copy()
        
        # Draw detections
        for det in self.latest_detections:
            bbox = det['bbox']
            confidence = det['confidence']
            distance = det.get('distance_cm')
            
            # Draw bounding box
            cv2.rectangle(
                vis_image, 
                (bbox[0], bbox[1]), 
                (bbox[2], bbox[3]), 
                (0, 255, 0),  # Green
                2
            )
            
            # Draw filled mask with transparency
            mask = det['mask']
            colored_mask = np.zeros_like(vis_image)
            colored_mask[:, :] = (0, 255, 0)  # Green overlay
            vis_image = np.where(
                mask[..., None] > 0,
                cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0),
                vis_image
            )
            
            # Draw label with distance
            if distance is not None:
                label = f"{det.get('class_name', det['id'])}: {confidence:.2f} ({distance:.1f}cm)"
            else:
                label = f"{det.get('class_name', det['id'])}: {confidence:.2f}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Label background
            cv2.rectangle(
                vis_image,
                (bbox[0], bbox[1] - label_size[1] - 10),
                (bbox[0] + label_size[0], bbox[1]),
                (0, 255, 0),
                -1
            )
            
            # Label text
            cv2.putText(
                vis_image,
                label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black text
                2
            )
        
        # Add info overlay
        mode_text = "CONTINUOUS" if self.continuous_detection else "SINGLE SHOT"
        info_text = f"Mode: {mode_text} | Objects: {len(self.latest_detections)}"
        
        cv2.putText(
            vis_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # White
            2
        )
        
        cv2.putText(
            vis_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),  # Black outline
            4
        )
        
        # Show image
        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    # Check for single-shot mode flag
    single_shot = '--single' in sys.argv
    
    try:
        node = SimpleSAMDetector(single_shot_mode=single_shot)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
