#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from custom_interfaces.srv import PixelToReal

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import numpy as np


class PixelToRealNode(Node):

    def __init__(self):
        super().__init__("pixel_to_real_node")

        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth/image_rect_raw",
            self.depth_callback,
            10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.info_callback,
            10
        )

        self.service = self.create_service(
            PixelToReal,
            "/pixel_to_real",
            self.handle_pixel_to_real
        )

        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()

        self.depth_image = None
        self.camera_ready = False

        # ---- DEPTH SCALE ----
        self.depth_scale = 0.001  # mm to meters

        # ---- Camera Pose in base_link ----
        self.t_base_cam = np.array([-0.109, 0.451, 0.66]) #0.0027, 0.5442, 0.6711
# 0.6371
# -0.0109, 0.5429, 0.6701


        self.R_base_cam = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ])

        self.get_logger().info("PixelToReal with TF ready.")

    # --------------------------------------------------

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def info_callback(self, msg):
        self.cam_model.fromCameraInfo(msg)
        self.camera_ready = True

    # --------------------------------------------------

    def get_robust_depth(self, u, v, window_size=5):

        h, w = self.depth_image.shape
        half = window_size // 2

        u_min = max(u - half, 0)
        u_max = min(u + half + 1, w)
        v_min = max(v - half, 0)
        v_max = min(v + half + 1, h)

        window = self.depth_image[v_min:v_max, u_min:u_max]
        valid = window[window > 0]

        if len(valid) == 0:
            return None

        return np.median(valid)

    # --------------------------------------------------

    def handle_pixel_to_real(self, request, response):

        if self.depth_image is None or not self.camera_ready:
            return response

        u = request.u
        v = request.v

        depth_raw = self.get_robust_depth(u, v)

        if depth_raw is None:
            self.get_logger().warn("No valid depth.")
            return response

        Z = float(depth_raw) * self.depth_scale

        # ============================================================
        # METHOD 1: Intrinsics-based (camera calibration dependent)
        # ============================================================
        # Camera frame 3D using PinholeCameraModel
        ray = self.cam_model.projectPixelTo3dRay((u, v))
        X_cam = ray[0] * Z
        Y_cam = ray[1] * Z
        Z_cam = Z
        p_cam = np.array([X_cam, Y_cam, Z_cam])
        
        # Transform to base frame
        p_base = self.R_base_cam @ p_cam + self.t_base_cam
        
        # ============================================================
        # METHOD 2: Empirical calibration (refined via least squares fit)
        # ============================================================
        # Fitted from 8 calibration points across the image
        # Coefficients derived from least-squares regression:
        # X = -0.433485 + 0.001057*u + 0.000123*v
        # Y = +0.811351 + 0.000001*u - 0.001093*v
        # RMS error: 0.0170 m (~17 mm)
        
        x_calib = -0.433485 + 0.001057 * u + 0.000123 * v
        y_calib = 0.811351 + 0.000001 * u - 0.001093 * v
        z_calib = 0.8  # Assume table height for now
        
        # Use empirical calibration (METHOD 2) - much more accurate
        response.x = float(x_calib)
        response.y = float(y_calib)
        response.z = float(z_calib)
        
        # Alternative intrinsics-based approach (commented out):
        # response.x = float(p_base[0])
        # response.y = float(p_base[1])
        # response.z = float(p_base[2])

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PixelToRealNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()