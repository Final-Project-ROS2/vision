
#!/usr/bin/env python3
"""
ROS2 RealSense Pixel-to-Real-World Service Node

This node provides a ROS2 service to convert pixel coordinates (u, v) to real-world 
3D coordinates (x, y, z) using Intel RealSense depth camera with proper coordinate transformation.

Service: /pixel_to_real
Type: custom_interfaces/srv/PixelToReal

Coordinate System:
- Camera center (depth cam): (0, -0.5442, 0.6711) meters in base frame
- Base/Table origin: (0, 0, 0)
- Floor: (0, 0, -0.805)
- Working area: x ∈ [-0.5, 0.5]m, y ∈ [-0.7, 0.0]m
- Image: 640x480 pixels (u ∈ [0, 640], v ∈ [0, 480])
- Table is planar at z = 0

Features:
- Aligns depth frames to color frames for accurate depth at pixel locations
- Uses RealSense intrinsics for proper 3D deprojection
- Transforms camera coordinates to robot base frame
- Validates output is within working area constraints
- Handles invalid depth with neighborhood search

Setup:
    ros2 run vision pixel_to_real_world_service

Usage:
    ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

try:
    from custom_interfaces.srv import PixelToReal
except ImportError:
    PixelToReal = None
    print("Warning: custom_interfaces.srv.PixelToReal not found. Build custom_interfaces package first.")


class PixelToRealWorldService(Node):
    """ROS2 service node for converting pixel coordinates to real-world 3D coordinates."""
    
    def __init__(self):
        super().__init__('pixel_to_real_node')
        
        
        self.rgb_topic = '/camera/color/image_raw'
        self.depth_topic = '/camera/depth/image_rect_raw'
        self.camera_info_topic = '/camera/color/camera_info'  # Use depth camera_info
        self.color_encoding = 'passthrough'
        self.depth_32_encoding = 'passthrough'
        self.depth_16_encoding = 'passthrough'

        
        # CvBridge for ROS<->OpenCV conversion
        self.bridge = CvBridge()
        
        # Latest frames from topics
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        self.depth_timestamp = None
        
        # Subscribe to camera topics
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.info_callback, 10)
        
        # Publisher for debug visualization
        self.debug_pub = self.create_publisher(Image, '/pixel_to_real_world/debug_image', 10)
        
        # Camera position in base frame (meters)
        self.cam_x_base = 0.0
        self.cam_y_base = -0.5442
        self.cam_z_base =  0.6711
        
        # Working area constraints (meters)
        self.x_min = -0.50
        self.x_max = 0.50
        self.y_min = -0.90
        self.y_max = 0.0
        self.z_table = 0.0  # Table height
        
        # Depth range for filtering (in meters)
        self.depth_min = 0.20
        self.depth_max = 0.6711
        
        self.get_logger().info(f'Camera position in base frame: ({self.cam_x_base}, {self.cam_y_base}, {self.cam_z_base})')
        self.get_logger().info(f'Working area: x=[{self.x_min}, {self.x_max}], y=[{self.y_min}, {self.y_max}]')
        self.get_logger().info(f'Depth range: {self.depth_min}m to {self.depth_max}m')
        self.get_logger().info(f'RGB topic: {self.rgb_topic}')
        self.get_logger().info(f'Depth topic: {self.depth_topic}')
        self.get_logger().info(f'Camera info topic: {self.camera_info_topic}')
        
        # Create ROS2 service
        if PixelToReal is not None:
            self.srv = self.create_service(
                PixelToReal,
                '/pixel_to_real',
                self.handle_pixel_to_real
            )
            self.get_logger().info('Service /pixel_to_real is ready')
            self.get_logger().info('Usage: ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"')
        else:
            self.get_logger().error('PixelToReal service type not available. Build custom_interfaces package.')
            raise RuntimeError('custom_interfaces.srv.PixelToReal not available')
    
    def rgb_callback(self, msg):
        """Store latest RGB image from topic."""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.color_encoding)
        except Exception as e:
            self.get_logger().error(f'Error converting RGB image: {e}')
    
    def depth_callback(self, msg):
        """Store latest depth image from topic."""
        try:
            # Convert depth image - handle both 16UC1 and 32FC1 formats
            if msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # Convert from millimeters to meters for 16UC1
                self.latest_depth = depth_image.astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.depth_encoding)
                if depth_image.dtype == np.uint16:
                    self.latest_depth = depth_image.astype(np.float32) / 1000.0
                else:
                    self.latest_depth = depth_image
            self.depth_timestamp = self.get_clock().now()
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
    
    def info_callback(self, msg):
        """Store camera intrinsics from CameraInfo topic."""
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info(f'Camera intrinsics received: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, '
                                  f'cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}')
            self.get_logger().info(f'Image size: {msg.width}x{msg.height}')
    
    def get_depth_at_pixel(self, u, v):
        """
        Get depth value at pixel coordinate with neighborhood fallback.
        
        Args:
            u, v: Pixel coordinates
            
        Returns:
            float: Depth in meters, or 0.0 if invalid
        """
        if self.latest_depth is None:
            self.get_logger().warn('No depth image available')
            return 0.0
        
        # Log depth image timestamp age
        if self.depth_timestamp is not None:
            age = (self.get_clock().now() - self.depth_timestamp).nanoseconds / 1e9
            self.get_logger().info(f'Depth image age: {age:.3f}s')
        
        h, w = self.latest_depth.shape
        if not (0 <= u < w and 0 <= v < h):
            self.get_logger().warn(f'Pixel ({u}, {v}) out of bounds ({w}x{h})')
            return 0.0
        
        # Get depth at pixel - read the ACTUAL current value
        depth = float(self.latest_depth[v, u])
        self.get_logger().info(f'Raw depth at pixel ({u}, {v}): {depth:.4f}m')
        
        # Validate depth
        if depth == 0 or depth < self.depth_min or depth > self.depth_max:
            self.get_logger().warn(f'Invalid depth at ({u}, {v}): {depth:.4f}m, searching neighborhood')
            depth = self._find_valid_depth_in_neighborhood(u, v)
        
        return depth
    
    def _find_valid_depth_in_neighborhood(self, u, v, window_size=5):
        """
        Find valid depth value in neighborhood of invalid pixel.
        
        Args:
            u, v: Pixel coordinates
            window_size: Radius of search window
            
        Returns:
            float: Median depth from valid neighbors, or 0.0 if none found
        """
        if self.latest_depth is None:
            return 0.0
        
        h, w = self.latest_depth.shape
        valid_depths = []
        
        for du in range(-window_size, window_size + 1):
            for dv in range(-window_size, window_size + 1):
                nu = u + du
                nv = v + dv
                
                if 0 <= nu < w and 0 <= nv < h:
                    depth = float(self.latest_depth[nv, nu])
                    if self.depth_min <= depth <= self.depth_max:
                        valid_depths.append(depth)
        
        if valid_depths:
            median_depth = float(np.median(valid_depths))
            self.get_logger().info(f'Found {len(valid_depths)} valid depths in neighborhood, median: {median_depth:.4f}m')
            return median_depth
        
        return 0.0
    
    def deproject_pixel_to_point(self, u, v, depth):
        """
        Deproject pixel to 3D point using camera intrinsics.
        
        Args:
            u, v: Pixel coordinates
            depth: Depth value in meters
            
        Returns:
            tuple: (x, y, z) in camera frame (meters)
        """
        if self.camera_info is None:
            self.get_logger().error('Camera info not available')
            return (0.0, 0.0, 0.0)
        
        # Extract intrinsics from camera info
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        
        # Deproject using pinhole camera model
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        
        # Store pixel coordinates for calibration correction later
        self._last_u = u
        self._last_v = v
        
        return (x_cam, y_cam, z_cam)
    
    def camera_to_base_transform(self, x_cam, y_cam, z_cam):
        """
        Transform coordinates from camera frame to robot base frame with empirical calibration.
        
        Calibration model derived from 9 ground truth measurements:
        - Y error has strong linear correlation with v-coordinate (vertical distortion)
        - X error correlates with u-coordinate (horizontal distortion)
        
        Regression analysis:
        Y_correction = -0.00116 * v + 0.164  (R² = 0.92)
        X_correction = -0.000194 * u + 0.042 (R² = 0.87)
        
        Args:
            x_cam: X coordinate in camera frame (meters)
            y_cam: Y coordinate in camera frame (meters)
            z_cam: Z coordinate in camera frame (meters)
            
        Returns:
            tuple: (x_base, y_base, z_base) in robot base frame (meters)
        """
        # Basic transformation
        x_base_raw = self.cam_x_base - x_cam
        y_base_raw = self.cam_y_base - y_cam
        z_base = self.cam_z_base - z_cam
        
        # Get pixel coordinates (stored during deprojection)
        u = getattr(self, '_last_u', 320)  # Default to center if not available
        v = getattr(self, '_last_v', 240)
        
        # Apply empirical calibration corrections based on pixel position
        # Updated 2026-02-25 using latest ground-truth pairs (u,v)->(x,y)
        # to minimize residual error in base frame outputs.
        y_correction = -0.00265 * v + 0.61800
        x_correction = -0.00060 * u + 0.22500
        
        # Apply corrections
        x_base = x_base_raw - x_correction
        y_base = y_base_raw - y_correction
        
        return (x_base, y_base, z_base)
    
    def validate_working_area(self, x, y, z):
        """
        Validate that the coordinates are within the working area.
        
        Args:
            x, y, z: Coordinates in base frame (meters)
            
        Returns:
            bool: True if within working area, False otherwise
        """
        if not (self.x_min <= x <= self.x_max):
            self.get_logger().warn(f'X={x:.4f}m out of range [{self.x_min}, {self.x_max}]')
            return False
        
        if not (self.y_min <= y <= self.y_max):
            self.get_logger().warn(f'Y={y:.4f}m out of range [{self.y_min}, {self.y_max}]')
            return False
        
        # Z can be above table (objects), just log info if too high or below table
        if z < self.z_table - 0.05:  # Below table surface
            self.get_logger().warn(f'Z={z:.4f}m is below table height {self.z_table}m')
            return False
        elif z > self.z_table + 0.70:  # Unreasonably high (>70cm above table)
            self.get_logger().warn(f'Z={z:.4f}m is unusually high above table')
        
        return True
    
    def pixel_to_3d_point(self, u, v):
        """
        Convert pixel coordinates (u, v) to 3D real-world coordinates in base frame.
        
        Uses topic-based depth reading and camera intrinsics for deprojection.
        
        Args:
            u: Pixel column (x-coordinate in image, 0-640)
            v: Pixel row (y-coordinate in image, 0-480)
            
        Returns:
            tuple: (x, y, z) in meters, robot base coordinate frame
        """
        # Check if we have camera data
        if self.latest_depth is None:
            self.get_logger().error('No depth data available')
            return (0.0, 0.0, 0.0)
        
        if self.camera_info is None:
            self.get_logger().error('No camera info available')
            return (0.0, 0.0, 0.0)
        
        # Get depth value at pixel
        depth_value = self.get_depth_at_pixel(int(u), int(v))
        
        self.get_logger().info(f'Depth at pixel ({u}, {v}): {depth_value:.4f}m')
        
        if depth_value == 0:
            self.get_logger().error(f'No valid depth found at or near pixel ({u}, {v})')
            return (0.0, 0.0, 0.0)
        
        # Deproject pixel to 3D point in camera coordinate frame
        point_3d_cam = self.deproject_pixel_to_point(float(u), float(v), depth_value)
        
        self.get_logger().info(f'Camera frame: x={point_3d_cam[0]:.4f}m, y={point_3d_cam[1]:.4f}m, z={point_3d_cam[2]:.4f}m')
        
        # Transform from camera frame to base frame
        x_base, y_base, z_base = self.camera_to_base_transform(
            point_3d_cam[0], 
            point_3d_cam[1], 
            point_3d_cam[2]
        )
        
        self.get_logger().info(f'Base frame: x={x_base:.4f}m, y={y_base:.4f}m, z={z_base:.4f}m')
        
        # Validate working area
        if not self.validate_working_area(x_base, y_base, z_base):
            self.get_logger().warn('Coordinates outside working area - returning anyway')
        
        # Clamp x and y to working area, but keep z_base as measured from depth
        # Don't clamp z - it represents the actual height of the object based on depth measurement
        x_base = np.clip(x_base, self.x_min, self.x_max)
        y_base = np.clip(y_base, self.y_min, self.y_max)
        # z_base is kept as-is from depth measurement
        
        return (float(x_base), float(y_base), float(z_base))
    
    def handle_pixel_to_real(self, request, response):
        """Handle service request to convert pixel to real-world coordinates."""
        u = int(request.u)
        v = int(request.v)
        
        self.get_logger().info(f'Received request: pixel ({u}, {v})')
        
        # Validate pixel coordinates
        if self.camera_info is not None:
            if not (0 <= u < self.camera_info.width and 0 <= v < self.camera_info.height):
                self.get_logger().error(f'Pixel ({u}, {v}) out of bounds. Image size: {self.camera_info.width}x{self.camera_info.height}')
                response.x = 0.0
                response.y = 0.0
                response.z = 0.0
                return response
        
        # Convert pixel to 3D point
        x, y, z = self.pixel_to_3d_point(u, v)

        # Invert axis to match robot's coordinate system
        x = -x
        y = -y 
        
        # Fill response
        response.x = x
        response.y = y
        response.z = z
        
        self.get_logger().info(f'Response: world coordinates (x={x:.4f}m, y={y:.4f}m, z={z:.4f}m)')
        
        # Publish debug visualization with red dot at requested pixel
        self._publish_debug_image(u, v, x, y, z)
        
        return response
    
    def _publish_debug_image(self, u, v, x, y, z):
        """
        Publish debug image with red dot at requested pixel location.
        
        Args:
            u, v: Pixel coordinates
            x, y, z: Computed world coordinates
        """
        if self.latest_rgb is None:
            return
        
        try:
            # Create a copy of the RGB image
            debug_image = self.latest_rgb.copy()
            
            # Ensure it's in BGR format for OpenCV
            if len(debug_image.shape) == 2:
                debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
            elif debug_image.shape[2] == 4:
                debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGBA2BGR)
            
            # Draw red circle at the requested pixel
            cv2.circle(debug_image, (u, v), 8, (0, 0, 255), -1)  # Filled red circle
            cv2.circle(debug_image, (u, v), 10, (0, 0, 255), 2)  # Red outline
            
            # Draw crosshair
            cv2.line(debug_image, (u-15, v), (u+15, v), (0, 0, 255), 2)
            cv2.line(debug_image, (u, v-15), (u, v+15), (0, 0, 255), 2)
            
            # Add text with coordinates
            text = f"({u},{v}) -> ({x:.3f}, {y:.3f}, {z:.3f})m"
            cv2.putText(debug_image, text, (u+15, v-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Convert back to ROS Image message
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = 'camera_color_optical_frame'
            
            # Publish
            self.debug_pub.publish(debug_msg)
            self.get_logger().info(f'Published debug image with red dot at ({u}, {v})')
            
        except Exception as e:
            self.get_logger().error(f'Error creating debug image: {e}')
    
    def destroy_node(self):
        """Clean up resources when node is destroyed."""
        self.get_logger().info('Shutting down /pixel_to_real service')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PixelToRealWorldService()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()