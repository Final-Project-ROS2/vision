#!/usr/bin/env python3
"""pixel_to_real_service.py

ROS2 Python node that provides a `/pixel_to_real` service.

Service Interface: `custom_interfaces.srv.PixelToReal`
  Request:
    int32 u     # pixel column (x-coordinate, positive to the right)
    int32 v     # pixel row (y-coordinate, positive downward)
  Response:
    float64 x   # world X coordinate (m, positive to the right)
    float64 y   # world Y coordinate (m, positive forward/away from camera) - with -0.722m offset applied
    float64 z   # world Z coordinate (m, height above table/ground)

Calibration Data:
  - Origin (0, 0, 0.8) in world: pixel (320, 500) - bottom center of image
  - Green box at world (0.5, 0, 0.8): pixel (320, 240)
  - Gear part at world (0.83, 0.03, 0.8): pixel (305, 95)
  - Drill at world (0.571546, -0.240961, 0.831898): pixel (466, 160.5)
  - Monkey wrench at world (0.623673, 0.372909, 0.806652): pixel (150, 200)
  - Table depth: 0.8 m from camera
  - Coordinate mapping:
    * u increases right → y DECREASES (u represents -y direction)
    * v increases down → x DECREASES (v represents -x direction)
    * x in world increases upward in image (opposite of v)
    * y in world increases leftward in image (opposite of u)

Setup:
  1. Build the custom_interfaces package:
     cd ~/final_project_ws
     colcon build --packages-select custom_interfaces
     source install/setup.bash
  
  2. Build the vision package:
     colcon build --packages-select vision --symlink-install
     source install/setup.bash
  
  3. Start the pixel_to_real service node:
     ros2 run vision pixel_to_real_service

Usage:
  # Example: green box at pixel (320, 240) → world (0.5, 0.0, 0.8)
  ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"
  
  # Example: gear part at pixel (305, 95) → world (0.83, 0.03, 0.8)
  ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: 305, v: 95}"
  
  # Example: drill at pixel (466, 160.5) → world (0.571546, -0.240961, 0.831898)
  ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: 466, v: 160}"
  
  # Example: monkey wrench at pixel (150, 200) → world (0.623673, 0.372909, 0.806652)
  ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: 150, v: 200}"
  
  # Example: origin at pixel (320, 500) → world (0.0, 0.0, 0.8)
  ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: 320, v: 500}"
  
  # View debug visualization:
  rqt_image_view /pixel_to_real/debug_image

How it works:
  Uses calibration points to compute a linear pixel-to-world transformation.
  Reads depth from /camera/depth/image_raw to determine z-coordinate.
  Shows debug visualization with pixel location and computed world coordinates.
"""


"""
# linear approx. formula
# Given pixel coordinates (u, v) and depth d from sensor:

# Step 1: Calculate pixel offset from origin
du = u - 320  # Origin u-coordinate
dv = v - 500  # Origin v-coordinate

# Step 2: Apply scaling and sign conversion
x = -dv * 0.001923  # scale_x = 0.5/260 ≈ 0.001923 m/pixel
y = -du * 0.002     # scale_y = 0.03/15 = 0.002 m/pixel

# Step 3: Convert depth to z-coordinate
z = 0.8 + (depth_reference - d)  # depth_reference set on first call


"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import math
import tf2_ros
import cv2
import pyrealsense2 as rs

# Replace with your actual service type. The repo often imports custom interfaces
# from `custom_interfaces.srv`. Adjust the import if your package name differs.
try:
    from custom_interfaces.srv import PixelToReal
except Exception:
    # The import may fail in editors; keep the name for when the package is built.
    PixelToReal = None


class PixelToRealServer(Node):
    def __init__(self,
                 rgb_topic: str = '/camera/image_raw',
                 depth_topic: str = '/camera/depth/image_raw',
                 info_topic: str = '/camera/camera_info',
                 default_target_frame: str = 'world'):
        super().__init__('pixel_to_real_server')

        # Parameter toggles simulated vs hardware camera topics
        self.declare_parameter('real_hardware', False)
        self.real_hardware = bool(self.get_parameter('real_hardware').value)

        # RealSense-specific variables for hardware mode
        self.rs_pipeline = None
        self.rs_align = None
        self.rs_intrinsics = None

        if self.real_hardware:
            self.rgb_topic = '/camera/color/image_raw'
            self.depth_topic = '/camera/depth/image_rect_raw'
            self.camera_info_topic = 'camera/color/camera_info'
            self.color_encoding = 'passthrough'
            self.depth_32_encoding = 'passthrough'
            self.depth_16_encoding = 'passthrough'
            
            # Initialize RealSense pipeline
            try:
                self.rs_pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                profile = self.rs_pipeline.start(config)
                
                # Create align object to align depth to color
                self.rs_align = rs.align(rs.stream.color)
                
                # Get intrinsics from color stream
                color_profile = profile.get_stream(rs.stream.color)
                self.rs_intrinsics = color_profile.as_video_stream_profile().intrinsics
                
                self.get_logger().info('RealSense pipeline initialized successfully')
                self.get_logger().info(f'Color intrinsics: fx={self.rs_intrinsics.fx}, fy={self.rs_intrinsics.fy}, ppx={self.rs_intrinsics.ppx}, ppy={self.rs_intrinsics.ppy}')
            except Exception as e:
                self.get_logger().error(f'Failed to initialize RealSense pipeline: {e}')
                self.get_logger().warn('Falling back to topic-based depth reading')
                self.rs_pipeline = None
        else:
            self.rgb_topic = rgb_topic or '/camera/image_raw'
            self.depth_topic = depth_topic or '/camera/depth/image_raw'
            self.camera_info_topic = info_topic or '/camera/camera_info'
            self.color_encoding = 'bgr8'
            self.depth_32_encoding = '32FC1'
            self.depth_16_encoding = '16UC1'

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_rgb_header = None
        self.latest_depth = None
        self.latest_depth_header = None
        self.camera_info = None
        self.default_target_frame = default_target_frame

        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_cb, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_cb, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.info_cb, 10)

        # Publisher for debug visualization
        self.debug_pub = self.create_publisher(Image, '/pixel_to_real/debug_image', 10)

        # Calibration data: pixel coordinates (u, v) -> world coordinates (x, y, z)
        # Using calibration points:
        # Origin: (u=320, v=500) -> (x=0, y=0, z=0.8)
        # Green box: (u=320, v=240) -> (x=0.5, y=0, z=0.8)
        # Gear: (u=305, v=95) -> (x=0.83, y=0.03, z=0.8)
        
        # Coordinate system mapping:
        # u increases right -> y DECREASES (u represents -y direction)
        # v increases down -> x DECREASES (v represents -x direction)
        
        self.u_origin = 320  # u=320 corresponds to y=0
        self.v_origin = 500  # v=500 corresponds to x=0
        
        # Calculate scaling factors from calibration points:
        # Green box: du=0, dv=-260 pixels -> dx=0.5, dy=0 meters
        # Gear: du=-15, dv=-405 pixels -> dx=0.83, dy=0.03 meters
        
        # From green box (vertical movement in image):
        # dv = 240 - 500 = -260 pixels (up in image)
        # dx = 0.5 - 0 = 0.5 meters (positive x, which is up)
        # scale_x = 0.5 / 260 = 0.00192 m/pixel
        
        # From gear (horizontal AND vertical movement):
        # du = 305 - 320 = -15 pixels (left in image)
        # dy = 0.03 - 0 = 0.03 meters (positive y, which is left)
        # scale_y = 0.03 / 15 = 0.002 m/pixel
        
        dv_green = 240 - 500  # -260 pixels (up in image)
        dx_green = 0.5 - 0     # 0.5 meters (positive x in world)
        
        du_gear = 305 - 320   # -15 pixels (left in image)
        dy_gear = 0.03 - 0    # 0.03 meters (positive y in world)
        
        self.scale_x = abs(dx_green / dv_green)  # 0.5/260 = 0.00192 m/pixel
        self.scale_y = abs(dy_gear / du_gear)    # 0.03/15 = 0.002 m/pixel
        
        # Depth calibration: Store reference depth for z-coordinate conversion
        # At calibration points, z should be 0.8m (table height)
        # We'll measure the actual depth sensor reading and use it as reference
        self.z_table = 0.8  # World z-coordinate of table surface
        self.depth_reference = None  # Will be set from first depth reading at calibration point
        
        self.get_logger().info(f'Pixel-to-world calibration: scale_x={self.scale_x:.6f} m/px, scale_y={self.scale_y:.6f} m/px')
        self.get_logger().info(f'Origin: pixel({self.u_origin}, {self.v_origin}) -> world(0, 0, 0.8)')
        self.get_logger().info(f'Coordinate mapping: u right=-y, v down=-x')
        self.get_logger().info(f'Calibrated from green box at (320,240)->(0.5,0,0.8) and gear at (305,95)->(0.83,0.03,0.8)')
        self.get_logger().info(f'Validation point: drill at (466,160)->(0.572,-0.241,0.832)')
        self.get_logger().info(f'Validation point: monkey_wrench at (150,200)->(0.624,0.373,0.807)')
        self.get_logger().info(f'Depth calibration: Call service at (320,240) to set depth reference for z=0.8m')
        self.get_logger().info(f'RGB topic: {self.rgb_topic}')
        self.get_logger().info(f'Depth topic: {self.depth_topic}')
        self.get_logger().info(f'Camera info topic: {self.camera_info_topic}')
        self.get_logger().info(f'real_hardware parameter: {self.real_hardware}')

        # Store calibration validation points for accuracy checking
        self.validation_points = [
            {"name": "green_box", "pixel": (320, 240), "world": (0.5, 0.0, 0.8)},
            {"name": "gear", "pixel": (305, 95), "world": (0.83, 0.03, 0.8)},
            {"name": "drill", "pixel": (466, 160), "world": (0.571546, -0.240961, 0.831898)},
            {"name": "monkey_wrench", "pixel": (150, 200), "world": (0.623673, 0.372909, 0.806652)},
            {"name": "origin", "pixel": (320, 500), "world": (0.0, 0.0, 0.8)}
        ]

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create service
        if PixelToReal is not None:
            self.srv = self.create_service(PixelToReal, 'pixel_to_real', self.handle_pixel_to_real)
            self.get_logger().info('Service /pixel_to_real ready (custom_interfaces.srv.PixelToReal)')
            self.get_logger().info('Usage: ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: 320, v: 240}"')
        else:
            self.get_logger().error('PixelToReal srv type not found!')
            self.get_logger().error('Build custom_interfaces: colcon build --packages-select custom_interfaces')
            raise RuntimeError('custom_interfaces.srv.PixelToReal not available')

    def rgb_cb(self, msg: Image):
        """Store the latest RGB image for pixel coordinate validation."""
        try:
            rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.color_encoding)
            self.latest_rgb = rgb_img
            self.latest_rgb_header = msg.header
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")

    def depth_cb(self, msg: Image):
        # Support 32FC1 and 16UC1 encodings; convert to float32 meters.
        try:
            if msg.encoding == '32FC1' or msg.encoding == '32F':
                depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.depth_32_encoding)
                depth = depth_img.astype(np.float32)
            elif msg.encoding == '16UC1' or msg.encoding == '16U':
                d16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.depth_16_encoding)
                depth = d16.astype(np.float32) / 1000.0  # assume mm -> m
            else:
                # Try a generic conversion to float32
                depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                depth = depth_img.astype(np.float32)

            self.latest_depth = depth
            self.latest_depth_header = msg.header
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def info_cb(self, msg: CameraInfo):
        self.camera_info = msg

    def pixel_to_world_calibrated(self, u: int, v: int, depth_m: float):
        """Convert pixel (u,v) to world coordinates (x,y,z) using calibration.
        
        Coordinate system:
        - u increases right -> y DECREASES (u represents -y direction)
        - v increases down -> x DECREASES (v represents -x direction)
        - Origin at pixel (320, 500) = world (0, 0, 0.8)
        - Depth is inversely related to z: small depth = high z (near camera, far from ground)
        
        Args:
            u: pixel column (positive right)
            v: pixel row (positive down)
            depth_m: depth in meters from camera (from depth sensor)
            
        Returns:
            (x, y, z) in world coordinates (meters)
        """
        # Calculate pixel offset from origin
        du = u - self.u_origin  # positive = right in image
        dv = v - self.v_origin  # positive = down in image
        
        # Apply transformation based on coordinate mapping:
        # v down (-dv up) -> x increases: x = -dv * scale_x
        # u right (-du left) -> y increases: y = -du * scale_y
        x = -dv * self.scale_x  # Up in image -> positive x
        y = -du * self.scale_y  # Left in image -> positive y
        
        # Apply y-offset for robot reference frame
        # y = y - 0.5442  # Shift y by -0.5442 meters
        
        # Convert depth to z-coordinate
        # Depth is inversely related to z: smaller depth = further from ground = higher z
        # At table (z=0.8), we need to calibrate based on actual depth reading
        # If depth_reference is set, use it; otherwise estimate from depth
        if self.real_hardware:
            z = depth_m  # Direct mapping for hardware
        elif self.depth_reference is not None:
            # z = z_table + (depth_reference - depth)
            # When depth < depth_reference (closer to camera), z increases
            # When depth > depth_reference (further from camera), z decreases
            z = self.z_table + (self.depth_reference - depth_m)
        else:
            # First call: assume this is close to table depth, set reference
            self.depth_reference = depth_m
            self.get_logger().info(f'Set depth reference: {self.depth_reference:.3f}m at z={self.z_table}m')
            z = self.z_table
        
        return (x, y, z)

    def read_depth_at(self, u: float, v: float, max_search: int = 5):
        """Read depth with bilinear interpolation; if invalid, search a median window.
        Returns depth in meters. Assumes table is at 0.8m depth.
        
        Handles invalid depth (NaN/zero/inf) common on reflective surfaces or sensor noise
        by searching a 5-pixel neighborhood and taking the median of valid depth values.
        
        The depth sensor returns distance from camera. We assume the table surface
        is at 0.8m from the camera, which should be the largest/most common depth value.
        """
        if self.latest_depth is None:
            self.get_logger().warn('No depth image available, using default table depth 0.8m')
            return 0.8  # Default table depth
        
        depth = self.latest_depth
        h, w = depth.shape[:2]
        if not (0 <= int(v) < h and 0 <= int(u) < w):
            return 0.8  # Default if out of bounds

        def bilinear(u_, v_):
            x0 = int(math.floor(u_)); x1 = min(x0 + 1, w - 1)
            y0 = int(math.floor(v_)); y1 = min(y0 + 1, h - 1)
            wa = (x1 - u_) * (y1 - v_)
            wb = (u_ - x0) * (y1 - v_)
            wc = (x1 - u_) * (v_ - y0)
            wd = (u_ - x0) * (v_ - y0)
            d00 = float(depth[y0, x0]); d10 = float(depth[y0, x1])
            d01 = float(depth[y1, x0]); d11 = float(depth[y1, x1])
            d = wa * d00 + wb * d10 + wc * d01 + wd * d11
            if np.isnan(d) or d <= 0.0 or np.isinf(d):
                return None
            return float(d)


        #ros2 service call /pixel_to_real custom_interfaces/srv/PixelToReal "{u: 220, v: 220}"
        # Try direct bilinear interpolation first
        if self.real_hardware:
            # Use RealSense SDK for proper depth reading
            if self.rs_pipeline is not None and self.rs_align is not None and self.rs_intrinsics is not None:
                try:
                    # Get frames from pipeline
                    frames = self.rs_pipeline.wait_for_frames(timeout_ms=1000)
                    
                    # Step 1: Align depth to color
                    aligned_frames = self.rs_align.process(frames)
                    
                    # Get aligned depth and color frames
                    aligned_depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    
                    if not aligned_depth_frame or not color_frame:
                        self.get_logger().warn('Failed to get aligned frames')
                        d = self.latest_depth[int(v), int(u)] if self.latest_depth is not None else 0.8
                    else:
                        # Step 2: Get depth at pixel (in meters)
                        depth_value = aligned_depth_frame.get_distance(int(u), int(v))
                        
                        # Step 3: Get intrinsics (already stored in self.rs_intrinsics)
                        # Step 4: Deproject pixel to 3D point
                        point_3d = rs.rs2_deproject_pixel_to_point(
                            self.rs_intrinsics,
                            [int(u), int(v)],
                            depth_value
                        )
                        
                        # point_3d is [x, y, z] in camera coordinate frame (meters)
                        # For this service, we primarily need the depth (z-component)
                        d = depth_value
                        
                        self.get_logger().info(f'Read depth at ({u},{v}): {d:.3f}m (RealSense SDK mode)')
                        self.get_logger().info(f'Deprojected 3D point in camera frame: x={point_3d[0]:.3f}, y={point_3d[1]:.3f}, z={point_3d[2]:.3f}')
                        
                        # Return the depth value for further processing
                        return d
                        
                except Exception as e:
                    self.get_logger().error(f'RealSense pipeline error: {e}')
                    # Fallback to topic-based depth
                    if self.latest_depth is not None:
                        d = self.latest_depth[int(v), int(u)]
                        self.get_logger().info(f'Fallback: Read depth at ({u},{v}): {d:.3f}m from topic')
                    else:
                        self.get_logger().warn('No depth data available, using default 0.8m')
                        return 0.8
            else:
                # Fallback to simple depth reading from topic
                if self.latest_depth is not None:
                    d = self.latest_depth[int(v), int(u)]
                    self.get_logger().info(f'Read depth at ({u},{v}): {d:.3f}m (topic-based hardware mode)')
                else:
                    self.get_logger().warn('No depth data available, using default 0.8m')
                    return 0.8









        else:
            d = bilinear(u, v)
            self.get_logger().info(f'Read depth at ({u},{v}): {d if d is not None else "invalid"} (simulated mode)')
        if d is not None:
            return d

        # Fallback: collect valid depths in neighborhood and take median
        valid_depths = []
        for du in range(-max_search, max_search + 1):
            for dv in range(-max_search, max_search + 1):
                uu = u + du
                vv = v + dv
                if 0 <= int(uu) < w and 0 <= int(vv) < h:
                    d_val = bilinear(uu, vv)
                    if d_val is not None:
                        valid_depths.append(d_val)
        
        if len(valid_depths) == 0:
            self.get_logger().warn(f'No valid depth found within {max_search}px of ({u:.1f},{v:.1f}), using table depth 0.8m')
            return 0.8  # Default table depth
        
        # Use median to be robust against outliers
        median_depth = float(np.median(valid_depths))
        return median_depth

    def backproject(self, u: float, v: float, d: float):
        # Use camera_info intrinsics
        K = self.camera_info.k
        fx = K[0]; fy = K[4]; cx = K[2]; cy = K[5]
        x_c = (u - cx) * d / fx
        y_c = (v - cy) * d / fy
        z_c = d
        return np.array([x_c, y_c, z_c], dtype=np.float64)

    def handle_pixel_to_real(self, req, resp):
        """Handle pixel to real coordinate conversion service request."""
        # Get pixel coordinates from request (int32 fields)
        u = int(req.u)
        v = int(req.v)

        # Get depth at this pixel (default to table depth 0.8m)
        depth_m = self.read_depth_at(float(u), float(v))
        
        # Convert pixel to world coordinates using calibration
        x_w, y_w, z_w = self.pixel_to_world_calibrated(u, v, depth_m)
        
        # DEBUG: Visualize the pixel location and world coordinates on the image
        if self.latest_rgb is not None:
            debug_img = self.latest_rgb.copy()
            
            # Draw a large crosshair at the requested pixel
            cv2.drawMarker(debug_img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
            
            # Draw a circle around it
            cv2.circle(debug_img, (u, v), 20, (0, 255, 0), 2)
            
            # Add text label with pixel coordinates
            label_pixel = f"Pixel: ({u}, {v})"
            cv2.putText(debug_img, label_pixel, (u + 25, v - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add text label with world coordinates
            label_world = f"World: ({x_w:.3f}, {y_w:.3f}, {z_w:.3f})m"
            cv2.putText(debug_img, label_world, (u + 25, v - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Add depth label
            label_depth = f"Depth: {depth_m:.3f}m"
            cv2.putText(debug_img, label_depth, (u + 25, v + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw origin marker at (320, 500)
            origin_u, origin_v = self.u_origin, self.v_origin
            if 0 <= origin_u < debug_img.shape[1] and 0 <= origin_v < debug_img.shape[0]:
                cv2.drawMarker(debug_img, (origin_u, origin_v), (255, 255, 0), 
                             cv2.MARKER_TILTED_CROSS, 30, 2)
                cv2.putText(debug_img, "Origin (0,0)", (origin_u + 10, origin_v - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw coordinate axes for reference
            cv2.putText(debug_img, "u -> (right)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(debug_img, "v", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(debug_img, "|", (10, 68),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(debug_img, "v (down)", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Publish debug image
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
                debug_msg.header.stamp = self.get_clock().now().to_msg()
                debug_msg.header.frame_id = 'camera_link'
                self.debug_pub.publish(debug_msg)
                self.get_logger().info(f'Published debug image for pixel ({u}, {v})')
            except Exception as e:
                self.get_logger().error(f'Failed to publish debug image: {e}')

        # Validate pixel coordinates against RGB image dimensions
        if self.latest_rgb is not None:
            rgb_h, rgb_w = self.latest_rgb.shape[:2]
            if not (0 <= u < rgb_w and 0 <= v < rgb_h):
                self.get_logger().error(f'Pixel ({u},{v}) out of RGB image bounds ({rgb_w}x{rgb_h})')
                # Return zero coordinates for out of bounds
                resp.x = 0.0
                resp.y = 0.0
                resp.z = 0.0
                return resp

        # Fill response (float64 fields)
        resp.x = float(x_w)
        resp.y = float(y_w)
        resp.z = float(z_w)
        
        # Check if this is a known calibration point and log the error
        for calib_point in self.validation_points:
            if abs(u - calib_point["pixel"][0]) <= 1 and abs(v - calib_point["pixel"][1]) <= 1:
                expected = calib_point["world"]
                error_x = x_w - expected[0]
                error_y = y_w - expected[1]
                error_z = z_w - expected[2]
                error_dist = np.sqrt(error_x**2 + error_y**2 + error_z**2)
                
                self.get_logger().info(
                    f'Validation point "{calib_point["name"]}" at pixel ({u},{v}): '
                    f'Calculated ({x_w:.3f}, {y_w:.3f}, {z_w:.3f}), '
                    f'Expected ({expected[0]:.3f}, {expected[1]:.3f}, {expected[2]:.3f}), '
                    f'Error: dx={error_x:.3f}m, dy={error_y:.3f}m, dz={error_z:.3f}m, dist={error_dist:.3f}m'
                )
                break
        
        self.get_logger().info(f'Pixel ({u},{v}) -> World ({x_w:.3f}m, {y_w:.3f}m, {z_w:.3f}m)')
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = PixelToRealServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
