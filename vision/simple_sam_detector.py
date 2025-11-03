#!/usr/bin/env python3
"""
Simple SAM Vision Detection Node
Focuses only on /camera/image_raw with OpenCV detection and visualization

Usage:
    ros2 run vision simple_sam_detector                    # Continuous mode
    ros2 run vision simple_sam_detector --single           # Single shot mode
    
Service:
    ros2 service call /vision/detect_objects std_srvs/srv/Trigger
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
from typing import List, Dict, Tuple


class SimpleSAMDetector(Node):
    """
    Simple SAM-based object detector
    
    Subscribes to:
        - /camera/image_raw (RGB images from Gazebo camera)
    
    Services:
        - /vision/detect_objects (Trigger detection on current image)
    
    Display:
        - Shows live camera feed with detections in OpenCV window
    """
    
    def __init__(self, single_shot_mode=False):
        super().__init__('simple_sam_detector')
        
        # Mode configuration
        self.single_shot_mode = single_shot_mode
        self.continuous_detection = not single_shot_mode
        
        # CV Bridge for ROS<->OpenCV conversion
        self.bridge = CvBridge()
        
        # Latest image from camera
        self.latest_rgb = None
        self.latest_detections = []
        
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
        
        # Detection service
        self.detection_service = self.create_service(
            Trigger,
            '/vision/detect_objects',
            self.detect_service_callback
        )
        
        # OpenCV window setup
        self.window_name = "SAM Object Detection - /camera/image_raw"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        # Timer for continuous visualization (30 Hz)
        self.viz_timer = self.create_timer(0.033, self.visualization_callback)
        
        mode_str = "SINGLE SHOT" if self.single_shot_mode else "CONTINUOUS"
        self.get_logger().info(f"🚀 Simple SAM Detector Started [{mode_str} MODE]")
        self.get_logger().info(f"📡 Subscribing to: /camera/image_raw")
        self.get_logger().info(f"🔧 Service: /vision/detect_objects")
        self.get_logger().info(f"👁️  OpenCV Window: '{self.window_name}'")
        
        if self.single_shot_mode:
            self.get_logger().info("💡 Call service to detect: ros2 service call /vision/detect_objects std_srvs/srv/Trigger")
        else:
            self.get_logger().info("💡 Running continuous detection on every frame")
    
    def rgb_callback(self, msg: Image):
        """Handle incoming RGB images from /camera/image_raw"""
        try:
            # Convert ROS Image message to OpenCV format (BGR8)
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # In continuous mode, detect on every frame
            if self.continuous_detection:
                self.latest_detections = self._detect_objects(self.latest_rgb)
                
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
    
    def detect_service_callback(self, request, response):
        """Service callback for /vision/detect_objects"""
        try:
            if self.latest_rgb is None:
                response.success = False
                response.message = "No image available from /camera/image_raw"
                self.get_logger().warn("⚠️ No image received yet")
                return response
            
            self.get_logger().info("🔍 Running SAM detection...")
            
            # Run detection on current image
            self.latest_detections = self._detect_objects(self.latest_rgb)
            
            response.success = True
            response.message = f"Detected {len(self.latest_detections)} objects"
            
            self.get_logger().info(f"✅ Detection complete: {len(self.latest_detections)} objects found")
            
            # Print detection details
            for i, det in enumerate(self.latest_detections):
                bbox = det['bbox']
                self.get_logger().info(
                    f"   Object {i+1}: bbox=[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}], "
                    f"area={det['area']}, confidence={det['confidence']:.2f}"
                )
            
        except Exception as e:
            response.success = False
            response.message = f"Detection error: {e}"
            self.get_logger().error(f"❌ Detection error: {e}")
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
            
            detection = {
                "id": f"obj_{i}",
                "bbox": [x, y, x + w_box, y + h_box],
                "center": [x + w_box // 2, y + h_box // 2],
                "area": int(area),
                "confidence": float(confidence),
                "mask": mask,
                "contour": contour
            }
            
            detections.append(detection)
        
        return detections
    
    def visualization_callback(self):
        """Display camera feed with detections in OpenCV window"""
        if self.latest_rgb is None:
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
        vis_image = self.latest_rgb.copy()
        
        # Draw detections
        for det in self.latest_detections:
            bbox = det['bbox']
            confidence = det['confidence']
            
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
            
            # Draw label
            label = f"{det['id']}: {confidence:.2f}"
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
