#!/usr/bin/env python3
"""
ROS2 Camera Service Node
Opens webcam or depth camera and publishes images to ROS2 topics

This node handles camera hardware and provides image streams for the vision pipeline.
Supports RGB webcams, Intel RealSense, and other depth cameras.

Author: ROS2 Vision Pipeline Team
License: Apache-2.0
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger, SetBool
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Optional, Tuple
import time


class CameraServiceNode(Node):
    """
    ROS2 node for camera capture and streaming
    
    Publishes:
        - /camera/image_raw (RGB images)
        - /camera/depth/image_raw (Depth images, if available)
        - /camera/camera_info (Camera intrinsic parameters)
    
    Services:
        - /camera/start (Start camera streaming)
        - /camera/stop (Stop camera streaming)
        - /camera/reset (Reset camera connection)
        - /camera/set_resolution (Change camera resolution)
    """
    
    def __init__(self):
        super().__init__('camera_service')
        
        # CV Bridge for OpenCV <-> ROS message conversion
        self.bridge = CvBridge()
        
        # Camera state
        self.camera = None
        self.depth_camera = None
        self.is_streaming = False
        self.camera_type = None  # 'webcam', 'realsense', 'file'
        
        # Camera parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('camera_type', 'webcam')  # 'webcam', 'realsense', 'file'
        self.declare_parameter('image_file', '')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('auto_start', True)
        
        # Get parameters
        self.camera_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        self.camera_type = self.get_parameter('camera_type').get_parameter_value().string_value
        self.image_file = self.get_parameter('image_file').get_parameter_value().string_value
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value
        self.auto_start = self.get_parameter('auto_start').get_parameter_value().bool_value
        
        # Publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        # Services
        self.start_srv = self.create_service(Trigger, '/camera/start', self.start_callback)
        self.stop_srv = self.create_service(Trigger, '/camera/stop', self.stop_callback)
        self.reset_srv = self.create_service(Trigger, '/camera/reset', self.reset_callback)
        
        # Timer for publishing frames
        self.timer = None
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("Camera Service Node Started")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"Camera Type: {self.camera_type}")
        self.get_logger().info(f"Camera ID: {self.camera_id}")
        self.get_logger().info(f"Resolution: {self.width}x{self.height}")
        self.get_logger().info(f"FPS: {self.fps}")
        
        # Auto-start camera if enabled
        if self.auto_start:
            self.get_logger().info("Auto-start enabled, opening camera...")
            self._open_camera()
    
    def _open_camera(self) -> bool:
        """Open camera device"""
        try:
            if self.camera_type == 'webcam':
                return self._open_webcam()
            elif self.camera_type == 'realsense':
                return self._open_realsense()
            elif self.camera_type == 'file':
                return self._open_file()
            else:
                self.get_logger().error(f"Unknown camera type: {self.camera_type}")
                return False
        except Exception as e:
            self.get_logger().error(f"Failed to open camera: {e}")
            return False
    
    def _open_webcam(self) -> bool:
        """Open standard USB webcam"""
        self.get_logger().info(f"Opening webcam {self.camera_id}...")
        
        self.camera = cv2.VideoCapture(self.camera_id)
        
        if not self.camera.isOpened():
            self.get_logger().error(f"Failed to open webcam {self.camera_id}")
            return False
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify settings
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        
        self.get_logger().info(f"Webcam opened successfully")
        self.get_logger().info(f"  Actual resolution: {actual_width}x{actual_height}")
        self.get_logger().info(f"  Actual FPS: {actual_fps}")
        
        # Start streaming
        self.is_streaming = True
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self._publish_frame)
        
        return True
    
    def _open_realsense(self) -> bool:
        """Open Intel RealSense depth camera"""
        try:
            import pyrealsense2 as rs
        except ImportError:
            self.get_logger().error("pyrealsense2 not installed. Install with: pip install pyrealsense2")
            return False
        
        self.get_logger().info("Opening RealSense camera...")
        
        try:
            # Configure RealSense pipeline
            self.rs_pipeline = rs.pipeline()
            config = rs.config()
            
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, int(self.fps))
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, int(self.fps))
            
            # Start pipeline
            profile = self.rs_pipeline.start(config)
            
            # Get intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            self.rs_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            self.get_logger().info("RealSense camera opened successfully")
            self.get_logger().info(f"  Resolution: {self.rs_intrinsics.width}x{self.rs_intrinsics.height}")
            self.get_logger().info(f"  Focal length: fx={self.rs_intrinsics.fx:.1f}, fy={self.rs_intrinsics.fy:.1f}")
            
            # Start streaming
            self.is_streaming = True
            self.camera_type = 'realsense'
            timer_period = 1.0 / self.fps
            self.timer = self.create_timer(timer_period, self._publish_realsense_frame)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to open RealSense: {e}")
            return False
    
    def _open_file(self) -> bool:
        """Open image or video file"""
        if not self.image_file:
            self.get_logger().error("No image file specified. Set 'image_file' parameter.")
            return False
        
        self.get_logger().info(f"Opening file: {self.image_file}")
        
        # Check if video or image
        if self.image_file.endswith(('.mp4', '.avi', '.mov')):
            self.camera = cv2.VideoCapture(self.image_file)
            if not self.camera.isOpened():
                self.get_logger().error(f"Failed to open video file: {self.image_file}")
                return False
        else:
            # Static image
            frame = cv2.imread(self.image_file)
            if frame is None:
                self.get_logger().error(f"Failed to load image: {self.image_file}")
                return False
            self.static_image = frame
        
        self.get_logger().info("File opened successfully")
        
        # Start streaming
        self.is_streaming = True
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self._publish_frame)
        
        return True
    
    def _publish_frame(self):
        """Publish RGB frame from webcam or file"""
        if not self.is_streaming:
            return
        
        try:
            # Read frame
            if hasattr(self, 'static_image'):
                frame = self.static_image.copy()
                ret = True
            else:
                ret, frame = self.camera.read()
            
            if not ret or frame is None:
                self.get_logger().warn("Failed to read frame")
                return
            
            # Convert to ROS message
            rgb_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            rgb_msg.header.stamp = self.get_clock().now().to_msg()
            rgb_msg.header.frame_id = 'camera_link'
            
            # Publish RGB
            self.rgb_pub.publish(rgb_msg)
            
            # Publish camera info
            self._publish_camera_info(frame.shape[1], frame.shape[0])
            
        except Exception as e:
            self.get_logger().error(f"Error publishing frame: {e}")
    
    def _publish_realsense_frame(self):
        """Publish RGB-D frame from RealSense"""
        if not self.is_streaming:
            return
        
        try:
            # Wait for frames
            frames = self.rs_pipeline.wait_for_frames(timeout_ms=1000)
            
            # Get RGB and depth
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                self.get_logger().warn("Failed to get frames")
                return
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Get timestamp
            timestamp = self.get_clock().now().to_msg()
            
            # Publish RGB
            rgb_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
            rgb_msg.header.stamp = timestamp
            rgb_msg.header.frame_id = 'camera_link'
            self.rgb_pub.publish(rgb_msg)
            
            # Publish Depth
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='passthrough')
            depth_msg.header.stamp = timestamp
            depth_msg.header.frame_id = 'camera_link'
            self.depth_pub.publish(depth_msg)
            
            # Publish camera info
            self._publish_camera_info_realsense()
            
        except Exception as e:
            self.get_logger().error(f"Error publishing RealSense frame: {e}")
    
    def _publish_camera_info(self, width: int, height: int):
        """Publish camera intrinsic parameters"""
        cam_info = CameraInfo()
        cam_info.header.stamp = self.get_clock().now().to_msg()
        cam_info.header.frame_id = 'camera_link'
        
        cam_info.width = width
        cam_info.height = height
        
        # Approximate intrinsics (assuming standard webcam FOV ~60 degrees)
        fx = width / (2.0 * np.tan(np.radians(60.0) / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        
        cam_info.k = [fx, 0.0, cx,
                      0.0, fy, cy,
                      0.0, 0.0, 1.0]
        
        cam_info.p = [fx, 0.0, cx, 0.0,
                      0.0, fy, cy, 0.0,
                      0.0, 0.0, 1.0, 0.0]
        
        cam_info.distortion_model = 'plumb_bob'
        cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.camera_info_pub.publish(cam_info)
    
    def _publish_camera_info_realsense(self):
        """Publish RealSense camera intrinsics"""
        cam_info = CameraInfo()
        cam_info.header.stamp = self.get_clock().now().to_msg()
        cam_info.header.frame_id = 'camera_link'
        
        cam_info.width = self.rs_intrinsics.width
        cam_info.height = self.rs_intrinsics.height
        
        cam_info.k = [self.rs_intrinsics.fx, 0.0, self.rs_intrinsics.ppx,
                      0.0, self.rs_intrinsics.fy, self.rs_intrinsics.ppy,
                      0.0, 0.0, 1.0]
        
        cam_info.p = [self.rs_intrinsics.fx, 0.0, self.rs_intrinsics.ppx, 0.0,
                      0.0, self.rs_intrinsics.fy, self.rs_intrinsics.ppy, 0.0,
                      0.0, 0.0, 1.0, 0.0]
        
        cam_info.distortion_model = 'plumb_bob'
        cam_info.d = list(self.rs_intrinsics.coeffs)
        
        self.camera_info_pub.publish(cam_info)
    
    def start_callback(self, request, response):
        """Service callback to start camera streaming"""
        if self.is_streaming:
            response.success = True
            response.message = "Camera already streaming"
            return response
        
        if self._open_camera():
            response.success = True
            response.message = "Camera started successfully"
            self.get_logger().info("Camera streaming started")
        else:
            response.success = False
            response.message = "Failed to start camera"
            self.get_logger().error("Failed to start camera streaming")
        
        return response
    
    def stop_callback(self, request, response):
        """Service callback to stop camera streaming"""
        if not self.is_streaming:
            response.success = True
            response.message = "Camera already stopped"
            return response
        
        self.is_streaming = False
        
        if self.timer:
            self.timer.cancel()
            self.timer = None
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        if hasattr(self, 'rs_pipeline'):
            self.rs_pipeline.stop()
        
        response.success = True
        response.message = "Camera stopped successfully"
        self.get_logger().info("Camera streaming stopped")
        
        return response
    
    def reset_callback(self, request, response):
        """Service callback to reset camera connection"""
        self.get_logger().info("Resetting camera...")
        
        # Stop current streaming
        if self.is_streaming:
            self.stop_callback(request, response)
            time.sleep(0.5)
        
        # Restart camera
        if self._open_camera():
            response.success = True
            response.message = "Camera reset successfully"
            self.get_logger().info("Camera reset completed")
        else:
            response.success = False
            response.message = "Failed to reset camera"
            self.get_logger().error("Camera reset failed")
        
        return response
    
    def __del__(self):
        """Cleanup on node destruction"""
        if self.camera:
            self.camera.release()
        if hasattr(self, 'rs_pipeline'):
            self.rs_pipeline.stop()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = CameraServiceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
