#!/usr/bin/env python3
"""
Common ROS2 utilities for vision pipeline

Shared functionality for camera handling, ROS setup, and common operations
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Optional, Tuple


class VisionNodeBase(Node):
    """Base class for vision nodes with common functionality"""
    
    def __init__(self, node_name: str):
        super().__init__(node_name)
        
        # Camera configuration
        self.declare_parameter('real_hardware', False)
        self.real_hardware = self.get_parameter('real_hardware').value
        
        # Set topic names based on hardware
        if self.real_hardware:
            self.rgb_topic = '/camera/color/image_raw'
            self.depth_topic = '/camera/depth/image_rect_raw'
            self.camera_info_topic = '/camera/color/camera_info'
            self.desired_encoding = 'passthrough'
        else:
            self.rgb_topic = '/camera/image_raw'
            self.depth_topic = '/camera/depth/image_raw'
            self.camera_info_topic = '/camera/camera_info'
            self.desired_encoding = 'bgr8'
        
        # Common setup
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()
        
        # Camera data
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.camera_info: Optional[CameraInfo] = None
        
        # QoS profiles
        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.service_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
    
    def setup_camera_subscriptions(self):
        """Set up camera subscriptions"""
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            self.image_qos
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            self.image_qos
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            self.service_qos
        )
    
    def rgb_callback(self, msg: Image):
        """Handle RGB image messages"""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, self.desired_encoding)
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")
    
    def depth_callback(self, msg: Image):
        """Handle depth image messages"""
        try:
            if self.real_hardware:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Handle camera info messages"""
        self.camera_info = msg
    
    def get_image_size(self) -> Tuple[int, int]:
        """Get current image size (width, height)"""
        if self.latest_rgb is not None:
            h, w = self.latest_rgb.shape[:2]
            return w, h
        return 640, 480  # Default
    
    def has_camera_data(self) -> bool:
        """Check if we have valid camera data"""
        return self.latest_rgb is not None


class OpenCVWindow:
    """Helper class for OpenCV window management"""
    
    def __init__(self, window_name: str, width: int = 800, height: int = 600):
        self.window_name = window_name
        self.width = width
        self.height = height
        
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, width, height)
        except Exception as e:
            print(f"Failed to create window {window_name}: {e}")
    
    def show(self, image: np.ndarray):
        """Display image in window"""
        try:
            cv2.imshow(self.window_name, image)
            cv2.waitKey(1)
        except Exception as e:
            print(f"Failed to display image: {e}")
    
    def close(self):
        """Close the window"""
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass


def draw_bbox(image: np.ndarray, bbox: list, label: str = "", 
              confidence: float = 0.0, color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Draw bounding box on image
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        label: Optional label text
        confidence: Optional confidence score
        color: Box color (B, G, R)
    
    Returns:
        Image with drawn bbox
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw label if provided
    if label:
        label_text = f"{label}"
        if confidence > 0:
            label_text += f" ({confidence:.2f})"
        
        # Calculate label size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label_text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return image


def ensure_custom_interfaces():
    """Check if custom interfaces are available"""
    try:
        from custom_interfaces.srv import DetectObjects, DetectGrasps, DetectGraspBBox
        from custom_interfaces.msg import SAMDetections, SAMDetection, GraspPose
        return True
    except ImportError:
        return False


# Import availability flags
CUSTOM_INTERFACES_AVAILABLE = ensure_custom_interfaces()

try:
    import torch
    from PIL import Image as PILImage
    from transformers import CLIPModel, AutoProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

GRASPNET_AVAILABLE = False  # Set based on your GraspNet installation