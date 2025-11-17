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
        self.previous_detections = []  # For IoU tracking
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
        
        # Service client for CLIP filtered classification
        self.clip_filter_client = self.create_client(
            Trigger,
            '/vision/classify_bbox_filtered'
        )
        
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
            
            # Run detection on captured frame (with IoU tracking from previous frame)
            self.latest_detections = self._detect_objects(frame_to_use)
            
            # Store current detections as previous for next frame IoU calculation
            self.previous_detections = self.latest_detections.copy()
            
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
            
            self.get_logger().info("=" * 80)
            self.get_logger().info("Running SAM detection and CLIP classification...")
            self.get_logger().info("=" * 80)
            
            # Step 1: Run SAM detection on captured frame
            self.latest_detections = self._detect_objects(frame_to_use)
            self.get_logger().info(f"SAM detected {len(self.latest_detections)} objects")
            
            # Store current detections as previous for next frame IoU calculation
            self.previous_detections = self.latest_detections.copy()
            
            # Step 2: Publish detections to trigger CLIP auto-classification
            self._publish_detections_ros()
            self.get_logger().info("Published detections to /vision/sam_detections (CLIP will auto-classify)")
            
            # Step 3: Wait briefly for CLIP to process (give it time to classify)
            import time
            time.sleep(0.5)  # 500ms delay for CLIP processing
            
            # Step 4: Call classify_bbox_filtered service to get filtered classifications
            clip_classifications = {}
            if self.clip_filter_client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info("Calling /vision/classify_bbox_filtered service...")
                clip_request = Trigger.Request()
                
                try:
                    future = self.clip_filter_client.call_async(clip_request)
                    rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
                    
                    if future.result() is not None:
                        clip_response = future.result()
                        if clip_response.success:
                            # Parse JSON response
                            clip_data = json.loads(clip_response.message)
                            self.get_logger().info(f"CLIP classified {clip_data.get('filtered_regions', 0)} objects with confidence > 0.5")
                            
                            # Map region_id to label for easy lookup
                            for region in clip_data.get('regions', []):
                                region_id = region.get('region_id')
                                clip_classifications[region_id] = {
                                    'label': region.get('label'),
                                    'confidence': region.get('confidence'),
                                    'bbox': region.get('bbox')
                                }
                        else:
                            self.get_logger().warn(f"CLIP classification failed: {clip_response.message}")
                    else:
                        self.get_logger().warn("CLIP service call failed (no response)")
                except Exception as e:
                    self.get_logger().warn(f"CLIP service call exception: {e}")
            else:
                self.get_logger().warn("CLIP classify_bbox_filtered service not available (timeout)")
            
            # Step 5: Build parallel arrays for response (merge SAM + CLIP results)
            object_ids = []
            bbox_x1 = []
            bbox_y1 = []
            bbox_x2 = []
            bbox_y2 = []
            confidences = []
            distances_cm = []
            
            for idx, det in enumerate(self.latest_detections):
                # Check if we have CLIP classification for this region
                clip_info = clip_classifications.get(idx)
                
                if clip_info:
                    # Use CLIP label and confidence
                    object_ids.append(f"{clip_info['label']}_{idx}")
                    confidences.append(float(clip_info['confidence']))
                    self.get_logger().info(f"  Region {idx}: {clip_info['label']} (CLIP confidence: {clip_info['confidence']:.2f})")
                else:
                    # Use SAM generic label
                    object_ids.append(det['id'])
                    confidences.append(float(det['confidence']))
                
                bbox = det['bbox']
                bbox_x1.append(bbox[0])
                bbox_y1.append(bbox[1])
                bbox_x2.append(bbox[2])
                bbox_y2.append(bbox[3])
                
                # Add distance if available
                distance = det.get('distance_cm')
                distances_cm.append(float(distance) if distance is not None else -1.0)
            
            # Build IoU metrics arrays
            iou_scores = []
            is_stable_array = []
            total_iou = 0.0
            stable_count = 0
            
            self.get_logger().info(f"Building IoU arrays from {len(self.latest_detections)} detections")
            
            for idx, det in enumerate(self.latest_detections):
                iou = det.get('iou_score', 0.0)
                is_stable = det.get('is_stable', False)
                
                self.get_logger().info(f"  Detection {idx}: iou_score={iou}, is_stable={is_stable}")
                
                iou_scores.append(float(iou))
                is_stable_array.append(bool(is_stable))
                
                if iou > 0:
                    total_iou += iou
                if is_stable:
                    stable_count += 1
            
            self.get_logger().info(f"IoU arrays built: {len(iou_scores)} scores, {stable_count} stable")
            
            # Calculate aggregate metrics
            num_dets = len(self.latest_detections)
            average_iou = total_iou / num_dets if num_dets > 0 else 0.0
            stability_rate = stable_count / num_dets if num_dets > 0 else 0.0
            
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
            response.iou_scores = iou_scores
            response.is_stable = is_stable_array
            response.average_iou = float(average_iou)
            response.stable_count = int(stable_count)
            response.stability_rate = float(stability_rate)
            response.error_message = ""
            
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"Detection + Classification complete: {len(self.latest_detections)} objects")
            self.get_logger().info("=" * 80)
            
            # Print bounding boxes with labels
            self.get_logger().info("Results (SAM + CLIP):")
            for i in range(len(object_ids)):
                self.get_logger().info(
                    f"  {object_ids[i]}: bbox=[{bbox_x1[i]}, {bbox_y1[i]}, {bbox_x2[i]}, {bbox_y2[i]}], "
                    f"conf={confidences[i]:.2f}, dist={distances_cm[i]:.1f}cm"
                )
            self.get_logger().info("=" * 80)
            
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
            response.iou_scores = []
            response.is_stable = []
            response.average_iou = 0.0
            response.stable_count = 0
            response.stability_rate = 0.0
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
            
            # Filter by confidence threshold (only keep detections > 0.4)
            if confidence <= 0.4:
                continue
            
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
            
            # Calculate IoU with previous frame detections (for AP-style metric)
            iou_score = 0.0
            matched_prev_id = None
            if self.previous_detections:
                best_iou = 0.0
                for prev_det in self.previous_detections:
                    iou = self._calculate_iou([x, y, x + w_box, y + h_box], prev_det['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        matched_prev_id = prev_det['id']
                iou_score = best_iou
            
            detection = {
                "id": f"obj_{i}",
                "class_name": "object",  # Generic class, can be enhanced with actual classification
                "confidence": float(confidence),
                "bbox": [x, y, x + w_box, y + h_box],
                "center": [center_x, center_y],
                "area": int(area),
                "distance_cm": distance_cm,
                "mask": mask,
                "contour": contour,
                "iou_score": float(iou_score),  # IoU with previous frame
                "matched_prev_id": matched_prev_id,
                "is_stable": iou_score >= 0.5  # COCO AP threshold (IoU >= 0.5)
            }
            
            detections.append(detection)
        
        return detections
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU score (0.0 to 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection rectangle
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if there is an intersection
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        if union_area == 0:
            return 0.0
        
        iou = intersection_area / union_area
        return float(iou)
    
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
        total_iou = 0.0
        stable_detections = 0  # COCO AP style: IoU >= 0.5
        
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
            
            # Add IoU metrics (COCO AP style)
            detection_obj["iou_with_previous"] = round(det.get("iou_score", 0.0), 3)
            detection_obj["is_stable_detection"] = det.get("is_stable", False)
            
            if det.get("iou_score", 0.0) > 0:
                total_iou += det["iou_score"]
            
            if det.get("is_stable", False):
                stable_detections += 1
            
            detections_list.append(detection_obj)
        
        # Calculate average distance
        average_distance = round(total_distance / distance_count, 1) if distance_count > 0 else None
        
        # Calculate COCO AP style metrics
        num_detections = len(self.latest_detections)
        average_iou = round(total_iou / num_detections, 3) if num_detections > 0 else 0.0
        stability_rate = round(stable_detections / num_detections, 3) if num_detections > 0 else 0.0
        
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
            },
            "metrics": {
                "coco_ap_style": {
                    "description": "IoU-based detection stability (frame-to-frame tracking)",
                    "average_iou": average_iou,
                    "stable_detections_count": stable_detections,
                    "stability_rate": stability_rate,
                    "iou_threshold": 0.5,
                    "note": "Stable detection = IoU >= 0.5 with previous frame (similar to COCO AP@0.5)"
                },
                "circularity_confidence": {
                    "description": "Shape-based confidence (geometric quality)",
                    "average_confidence": round(sum([d["confidence"] for d in detections_list]) / num_detections, 3) if num_detections > 0 else 0.0
                }
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
                
                # IoU tracking fields (COCO AP-style metrics)
                sam_det.iou_score = float(det.get('iou_score', 0.0))
                sam_det.matched_prev_id = str(det.get('matched_prev_id', ''))
                sam_det.is_stable_detection = bool(det.get('is_stable', False))
                
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
        for idx, det in enumerate(self.latest_detections):
            bbox = det['bbox']
            confidence = det['confidence']
            distance = det.get('distance_cm')
            obj_no = idx  # Object number
            
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
            
            # Draw label with object number and distance
            if distance is not None:
                label = f"#{obj_no} {det.get('class_name', det['id'])}: {confidence:.2f} ({distance:.1f}cm)"
            else:
                label = f"#{obj_no} {det.get('class_name', det['id'])}: {confidence:.2f}"
            
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
