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

Calibration Method (iLogic Hybrid):
  Linear model fitted from 22 empirical measurements (least-squares, LOO-CV RMSE ≈ 2.0 cm):
    x =  0.00130317·u + 0.00002114·v − 0.56859693
    y = −0.00002728·u − 0.00133088·v + 0.98011251

  A Gaussian-IDW empirical correction is added on top:
    - Problem Zone (near any of the 22 samples): correction applied → RMSE ≈ 1.5 cm
    - Golden Zone  (far from all samples):       pure linear model used → avoids overfitting

  Empirical samples (u, v → world x, y):
    (560,362)→(0.165,0.463)  (468,452)→(0.064,0.367)
    (334,336)→(-0.135,0.495) (241,432)→(-0.245,0.405)
    (598,245)→(0.205,0.618)  (493,342)→(0.104,0.520)
    (304,327)→(-0.183,0.510) (206,423)→(-0.295,0.418)
    (587,113)→(0.195,0.810)  (490,202)→(0.090,0.715)
    (273,111)→(-0.220,0.800) (177,206)→(-0.325,0.705)
    (308,302)→(-0.164,0.572) (276,324)→(-0.200,0.533)
    (555,298)→(0.145,0.576)  (456,384)→(0.040,0.482)
    (343,284)→(-0.138,0.580) (247,375)→(-0.240,0.485)
    (562, 85)→(0.166,0.860)  (468,173)→(0.065,0.770)
    (202, 98)→(-0.310,0.835) (101,196)→(-0.415,0.740)

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
            
            # Note: We DO NOT initialize RealSense pipeline here!
            # The camera is published by a separate node (e.g., realsense-ros)
            # We only subscribe to the published topics
            self.get_logger().info('Real hardware mode: subscribing to RealSense topics')
            self.get_logger().info(f'  RGB: {self.rgb_topic}')
            self.get_logger().info(f'  Depth: {self.depth_topic}')
            self.get_logger().info(f'  Camera Info: {self.camera_info_topic}')
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

        # ── iLogic Calibration ────────────────────────────────────────────────
        # Linear model fitted from 22 empirical measurements (least-squares):
        #   x =  0.00130317 * u  +0.00002114 * v  -0.56859693
        #   y = -0.00002728 * u  -0.00133088 * v  +0.98011251
        # LOO-CV RMSE ≈ 0.020 m
        self.lin_cx = np.array([+0.00130317, +0.00002114, -0.56859693])  # [u, v, 1] → x
        self.lin_cy = np.array([-0.00002728, -0.00133088, +0.98011251])  # [u, v, 1] → y

        # Empirical correction table  (u, v, residual_x, residual_y)
        # residual = true_world - linear_prediction  ← precomputed offline
        # Used by the IDW Gaussian kernel to correct systematic lens distortion.
        self._emp = np.array([
            # u     v      res_x      res_y
            [560,  362, -0.00383,  -0.02010],  # M1G_TR
            [468,  452,  0.01316,   0.00117],  # M1G_BL
            [334,  336, -0.00878,  -0.02882],  # M1P_TR
            [241,  432,  0.00040,   0.00640],  # M1P_BL
            [598,  245, -0.01089,  -0.01965],  # M2G_TR
            [493,  342,  0.02291,   0.00854],  # M2G_BL
            [304,  327, -0.01747,  -0.02659],  # M2P_TR
            [206,  423, -0.00384,   0.00649],  # M2P_BL
            [587,  113, -0.00378,  -0.00365],  # M3G_TR
            [490,  202,  0.01584,   0.01714],  # M3G_BL
            [273,  111, -0.00952,  -0.02493],  # M3P_TR
            [177,  206,  0.00856,   0.00391],  # M3P_BL
            [308,  302, -0.00316,   0.00224],  # M3X_TR
            [276,  324,  0.00207,  -0.00836],  # M3X_BL
            [555,  298, -0.01601,   0.00758],  # M4G_TR
            [456,  384,  0.00623,   0.02542],  # M4G_BL
            [343,  284, -0.02237,  -0.01282],  # M4P_TR
            [247,  375, -0.00124,   0.01070],  # M4P_BL
            [562,   85,  0.00041,   0.00834],  # M5G_TR
            [468,  173,  0.02007,   0.03286],  # M5G_BL
            [202,   98, -0.00669,  -0.00916],  # M5P_TR
            [101,  196,  0.01778,   0.02352],  # M5P_BL
        ], dtype=np.float64)
        # Gaussian kernel bandwidth (pixels). Controls how far correction influence spreads.
        self._idw_sigma = 80.0
        # Threshold: if IDW weight-sum < this fraction of max possible, treat as Golden Zone.
        self._golden_weight_threshold = 0.05

        # Depth calibration
        self.z_table = 0.8
        self.depth_reference = None

        self.get_logger().info('iLogic pixel-to-world: linear model + IDW empirical correction')
        self.get_logger().info(f'  x = {self.lin_cx[0]:+.8f}*u {self.lin_cx[1]:+.8f}*v {self.lin_cx[2]:+.8f}')
        self.get_logger().info(f'  y = {self.lin_cy[0]:+.8f}*u {self.lin_cy[1]:+.8f}*v {self.lin_cy[2]:+.8f}')
        self.get_logger().info(f'  IDW sigma={self._idw_sigma}px  empirical samples={len(self._emp)}')
        self.get_logger().info(f'RGB topic: {self.rgb_topic}')
        self.get_logger().info(f'Depth topic: {self.depth_topic}')
        self.get_logger().info(f'Camera info topic: {self.camera_info_topic}')
        self.get_logger().info(f'real_hardware parameter: {self.real_hardware}')

        # Store calibration validation points for accuracy checking
        self.validation_points = [
            {"name": "M1G_TR",       "pixel": (560, 362), "world": ( 0.165,  0.463, 0.8)},
            {"name": "M1P_TR",       "pixel": (334, 336), "world": (-0.135,  0.495, 0.8)},
            {"name": "M3G_TR",       "pixel": (587, 113), "world": ( 0.195,  0.810, 0.8)},
            {"name": "M3P_TR",       "pixel": (273, 111), "world": (-0.220,  0.800, 0.8)},
            {"name": "M5G_TR",       "pixel": (562,  85), "world": ( 0.166,  0.860, 0.8)},
            {"name": "M5P_TR",       "pixel": (202,  98), "world": (-0.310,  0.835, 0.8)},
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

    def _idw_correction(self, u: float, v: float):
        """iLogic: compute Gaussian-IDW empirical correction at pixel (u, v).

        Golden Zone  (low influence from empirical samples) → correction ≈ 0.
        Problem Zone (high influence near known distortion samples) → correction applied.

        Returns (corr_x, corr_y, weight_sum_norm) where weight_sum_norm in [0, 1].
        """
        eu = self._emp[:, 0]
        ev = self._emp[:, 1]
        err_x = self._emp[:, 2]
        err_y = self._emp[:, 3]

        dists = np.sqrt((eu - u) ** 2 + (ev - v) ** 2)
        weights = np.exp(-0.5 * (dists / self._idw_sigma) ** 2)
        w_sum = weights.sum()

        # Maximum possible weight_sum (if u,v were exactly on a sample point)
        max_w_sum = len(self._emp) * 1.0  # upper bound: all weights=1
        w_norm = w_sum / max_w_sum

        if w_sum < 1e-12:
            return 0.0, 0.0, 0.0

        corr_x = float(np.dot(weights, err_x) / w_sum)
        corr_y = float(np.dot(weights, err_y) / w_sum)
        return corr_x, corr_y, w_norm

    def pixel_to_world_calibrated(self, u: int, v: int, depth_m: float):
        """iLogic pixel → world conversion.

        1. Apply the empirically fitted linear model (replaces the old 2-point
           scale_x / scale_y formula which had large systematic errors).
        2. Add a Gaussian-IDW empirical correction:
             - Problem Zone (high weight from nearby samples): correction applied fully.
             - Golden Zone  (low weight / far from all samples): correction fades to 0,
               pure linear model used — avoids overfitting in unsampled areas.

        Args:
            u: pixel column (positive right)
            v: pixel row (positive down)
            depth_m: depth in meters from camera

        Returns:
            (x, y, z) in world coordinates (meters)
        """
        # ── Step 1: Linear model (Golden Zone baseline) ───────────────────────
        feat = np.array([u, v, 1.0])
        x = float(np.dot(self.lin_cx, feat))
        y = float(np.dot(self.lin_cy, feat))

        # ── Step 2: iLogic empirical correction ───────────────────────────────
        corr_x, corr_y, w_norm = self._idw_correction(float(u), float(v))

        if w_norm >= self._golden_weight_threshold:
            # Problem Zone: apply correction
            x += corr_x
            y += corr_y
            zone = 'problem'
        else:
            # Golden Zone: trust linear model, skip correction
            zone = 'golden'

        self.get_logger().debug(
            f'iLogic zone={zone} w_norm={w_norm:.3f} '  
            f'corr=({corr_x:+.4f},{corr_y:+.4f}) '  
            f'-> x={x:.4f} y={y:.4f}'
        )

        # ── Step 3: Depth → z ─────────────────────────────────────────────────
        if self.real_hardware:
            z = depth_m
        elif self.depth_reference is not None:
            z = self.z_table + (self.depth_reference - depth_m)
        else:
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
        # Use topic-based depth reading for both hardware and simulation
        if self.latest_depth is not None:
            d = self.latest_depth[int(v), int(u)]
            self.get_logger().info(f'Read depth at ({u},{v}): {d:.3f}m from topic')
        else:
                    self.get_logger().warn('No depth data available, using default 0.8m')
                    return 0.8








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
            
            # Draw frame-centre marker (approximate optical centre)
            origin_u, origin_v = 320, 240
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
