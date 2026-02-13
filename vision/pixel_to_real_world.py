
#!/usr/bin/env python3
"""
ROS2 RealSense Pixel-to-Real-World Service Node

This node provides a ROS2 service to convert pixel coordinates (u, v) to real-world 
3D coordinates (x, y, z) using Intel RealSense depth camera with proper coordinate transformation.

Service: /pixel_to_real_world
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
import pyrealsense2 as rs
import numpy as np
import cv2

try:
    from custom_interfaces.srv import PixelToReal
except ImportError:
    PixelToReal = None
    print("Warning: custom_interfaces.srv.PixelToReal not found. Build custom_interfaces package first.")


class PixelToRealWorldService(Node):
    """ROS2 service node for converting pixel coordinates to real-world 3D coordinates."""
    
    def __init__(self):
        super().__init__('pixel_to_real_world_service')
        
        # Camera position in base frame (meters)
        self.cam_x_base = 0.0
        self.cam_y_base = -0.5442
        self.cam_z_base = 0.6711
        
        # Working area constraints (meters)
        self.x_min = -0.50
        self.x_max = 0.50
        self.y_min = -0.90
        self.y_max = 0.0
        self.z_table = 0.0  # Table height
        
        self.get_logger().info(f'Camera position in base frame: ({self.cam_x_base}, {self.cam_y_base}, {self.cam_z_base})')
        self.get_logger().info(f'Working area: x=[{self.x_min}, {self.x_max}], y=[{self.y_min}, {self.y_max}]')
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable depth and color streams at 640x480
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline and get profile
        try:
            self.profile = self.pipeline.start(self.config)
            self.get_logger().info('RealSense pipeline started successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to start RealSense pipeline: {e}')
            raise
        
        # Get device and depth sensor information
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        
        # Get depth scale - converts raw depth units to meters
        self.depth_scale = depth_sensor.get_depth_scale()
        self.get_logger().info(f'Depth scale: {self.depth_scale}')
        
        # Set depth range for filtering (in meters)
        # Camera is at 0.6711m height, table at 0m, so distance to table is ~0.67m
        self.depth_min = 0.20  # minimum depth in meters
        self.depth_max = 1.50   # maximum depth in meters
        self.get_logger().info(f'Depth range: {self.depth_min}m to {self.depth_max}m')
        
        # Get intrinsics for depth and color streams
        self.depth_intrin = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.color_intrin = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        self.get_logger().info(f'Color intrinsics: fx={self.color_intrin.fx:.2f}, fy={self.color_intrin.fy:.2f}, '
                              f'ppx={self.color_intrin.ppx:.2f}, ppy={self.color_intrin.ppy:.2f}')
        self.get_logger().info(f'Image size: {self.color_intrin.width}x{self.color_intrin.height}')
        
        # Create align object to align depth frames to color frames
        self.align = rs.align(rs.stream.color)
        
        # Store latest frames
        self.latest_aligned_depth_frame = None
        self.latest_color_frame = None
        
        # Warm up the camera (skip first few frames)
        self.get_logger().info('Warming up camera...')
        for _ in range(30):
            self.pipeline.wait_for_frames()
        self.get_logger().info('Camera ready')
        
        # Create ROS2 service
        if PixelToReal is not None:
            self.srv = self.create_service(
                PixelToReal,
                'pixel_to_real_world',
                self.handle_pixel_to_real_world
            )
            self.get_logger().info('Service /pixel_to_real_world is ready')
            self.get_logger().info('Usage: ros2 service call /pixel_to_real_world custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"')
        else:
            self.get_logger().error('PixelToReal service type not available. Build custom_interfaces package.')
            raise RuntimeError('custom_interfaces.srv.PixelToReal not available')
    
    def get_latest_frames(self):
        """Capture and align the latest frames from RealSense camera."""
        try:
            # Wait for frames with timeout
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            
            # Align depth frame to color frame
            aligned_frames = self.align.process(frames)
            
            # Get aligned depth and color frames
            self.latest_aligned_depth_frame = aligned_frames.get_depth_frame()
            self.latest_color_frame = aligned_frames.get_color_frame()
            
            if not self.latest_aligned_depth_frame or not self.latest_color_frame:
                self.get_logger().warn('Failed to get valid frames')
                return False
            
            return True
        except Exception as e:
            self.get_logger().error(f'Error getting frames: {e}')
            return False
    
    def camera_to_base_transform(self, x_cam, y_cam, z_cam):
        """
        Transform coordinates from camera frame to robot base frame.
        
        Camera coordinate system (RealSense convention):
        - X: right (+u direction in image)
        - Y: down (-v direction in image, since v increases downward)
        - Z: forward (depth, away from camera towards table)
        
        Base coordinate system:
        - X: forward (+u direction)
        - Y: left (+y and -v are same direction)
        - Z: up
        
        Camera is positioned at (0, -0.5442, 0.6711) looking down at the table.
        
        Coordinate mapping:
        - +u (image right) → +x_cam (camera right) → +x_base (base forward)
        - -v (image up) → -y_cam (camera up) → +y_base (base left)
        - depth → +z_cam (camera forward/down) → projects onto table plane
        
        Args:
            x_cam: X coordinate in camera frame (meters) - right direction
            y_cam: Y coordinate in camera frame (meters) - down direction
            z_cam: Z coordinate in camera frame (meters) - depth/forward
            
        Returns:
            tuple: (x_base, y_base, z_base) in robot base frame (meters)
        """
        # Transform from camera frame to base frame
        # Camera is looking straight down at the table
        
        # +u (image right) → +x_cam (camera right) → +x_base (base forward)
        x_base = self.cam_x_base + x_cam
        
        # -v (image up) → -y_cam (camera up) → +y_base (base left)
        # So: +y_cam (camera down) → -y_base (base right)
        y_base = self.cam_y_base - y_cam
        
        # Depth (z_cam) determines the z position on table
        # Camera is at height 0.6711m, pointing down
        # z_cam is the distance from camera, so actual height is:
        z_base = self.cam_z_base - z_cam
        
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
        
        # Z should be close to table height (allow some tolerance)
        if abs(z - self.z_table) > 0.05:  # 5cm tolerance
            self.get_logger().warn(f'Z={z:.4f}m deviates from table height {self.z_table}m')
        
        return True
    
    def pixel_to_3d_point(self, u, v):
        """
        Convert pixel coordinates (u, v) to 3D real-world coordinates in base frame.
        
        Uses pyrealsense2 deprojection with aligned depth frames.
        
        Args:
            u: Pixel column (x-coordinate in image, 0-640)
            v: Pixel row (y-coordinate in image, 0-480)
            
        Returns:
            tuple: (x, y, z) in meters, robot base coordinate frame
        """
        # Get latest frames
        if not self.get_latest_frames():
            self.get_logger().error('Failed to get camera frames')
            return (0.0, 0.0, 0.0)
        
        # Get depth value at the pixel location in aligned depth frame
        depth_value = self.latest_aligned_depth_frame.get_distance(int(u), int(v))
        
        self.get_logger().info(f'Raw depth at pixel ({u}, {v}): {depth_value:.4f}m')
        
        # Validate depth
        if depth_value == 0 or depth_value < self.depth_min or depth_value > self.depth_max:
            self.get_logger().warn(f'Invalid depth at pixel ({u}, {v}): {depth_value:.4f}m (range: {self.depth_min}-{self.depth_max}m)')
            # Try to find valid depth in neighborhood
            depth_value = self._find_valid_depth_in_neighborhood(int(u), int(v))
            if depth_value == 0:
                self.get_logger().error(f'No valid depth found near pixel ({u}, {v})')
                return (0.0, 0.0, 0.0)
            self.get_logger().info(f'Using neighborhood depth: {depth_value:.4f}m')
        
        # Deproject pixel to 3D point in camera coordinate frame
        # Using color intrinsics since depth is aligned to color
        pixel = [float(u), float(v)]
        point_3d_cam = rs.rs2_deproject_pixel_to_point(self.color_intrin, pixel, depth_value)
        
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
        
        # Clamp to working area
        x_base = np.clip(x_base, self.x_min, self.x_max)
        y_base = np.clip(y_base, self.y_min, self.y_max)
        z_base = np.clip(z_base, self.z_table - 0.05, self.z_table + 0.05)
        
        return (float(x_base), float(y_base), float(z_base))
    
    def _find_valid_depth_in_neighborhood(self, u, v, window_size=5):
        """
        Find valid depth value in neighborhood of invalid pixel.
        
        Args:
            u, v: Pixel coordinates
            window_size: Radius of search window
            
        Returns:
            float: Median depth from valid neighbors, or 0.0 if none found
        """
        valid_depths = []
        
        for du in range(-window_size, window_size + 1):
            for dv in range(-window_size, window_size + 1):
                nu = u + du
                nv = v + dv
                
                if 0 <= nu < self.color_intrin.width and 0 <= nv < self.color_intrin.height:
                    depth = self.latest_aligned_depth_frame.get_distance(nu, nv)
                    if self.depth_min <= depth <= self.depth_max:
                        valid_depths.append(depth)
        
        if valid_depths:
            median_depth = float(np.median(valid_depths))
            self.get_logger().info(f'Found {len(valid_depths)} valid depths in neighborhood, median: {median_depth:.4f}m')
            return median_depth
        
        return 0.0
    
    def handle_pixel_to_real_world(self, request, response):
        """Handle service request to convert pixel to real-world coordinates."""
        u = int(request.u)
        v = int(request.v)
        
        self.get_logger().info(f'Received request: pixel ({u}, {v})')
        
        # Validate pixel coordinates
        if not (0 <= u < self.color_intrin.width and 0 <= v < self.color_intrin.height):
            self.get_logger().error(f'Pixel ({u}, {v}) out of bounds. Image size: {self.color_intrin.width}x{self.color_intrin.height}')
            response.x = 0.0
            response.y = 0.0
            response.z = 0.0
            return response
        
        # Convert pixel to 3D point
        x, y, z = self.pixel_to_3d_point(u, v)
        
        # Fill response
        response.x = x
        response.y = y
        response.z = z
        
        self.get_logger().info(f'Response: world coordinates (x={x:.4f}m, y={y:.4f}m, z={z:.4f}m)')
        
        return response
    
    def destroy_node(self):
        """Clean up resources when node is destroyed."""
        try:
            self.pipeline.stop()
            self.get_logger().info('RealSense pipeline stopped')
        except Exception as e:
            self.get_logger().error(f'Error stopping pipeline: {e}')
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