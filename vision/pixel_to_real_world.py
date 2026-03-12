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
        self.t_base_cam = np.array([-0.0361, 0.5303, 0.6458]) #0.0027, 0.5442, 0.6711
# 0.6371
# -0.0109, 0.5429, 0.6701
# -0.0386, 0.5303, 0.5238

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

    def _compute_intrinsic_estimate(self, u, v, Z):
        """
        METHOD 1: Intrinsics-based calibration using pinhole camera model.
        Uses camera intrinsics and known extrinsics (R, t).
        More physically grounded but depends on calibration accuracy.
        """
        ray = self.cam_model.projectPixelTo3dRay((u, v))
        X_cam = ray[0] * Z
        Y_cam = ray[1] * Z
        Z_cam = Z
        p_cam = np.array([X_cam, Y_cam, Z_cam])
        
        # Transform to base frame
        p_base = self.R_base_cam @ p_cam + self.t_base_cam
        
        return p_base

    def _compute_empirical_estimate(self, u, v):
        """
        METHOD 2: Empirical calibration using least-squares fitted coefficients.
        Coefficients fitted from 20 calibration points (10 measurements x 2 points each).
        Calibration date: March 12, 2026
        Camera coordinate: [-0.0361, 0.5303, 0.6458]
        Height: 0.67 m
        
        Fitted coefficients (RMS errors):
        X = -0.313151 + 0.000949*u - 0.000051*v   (RMS error: ~35.8 mm)
        Y = 0.821247 - 0.000062*u - 0.001042*v    (RMS error: ~5.84 mm)
        Overall 2D RMS error: ~36.27 mm
        """
        x_calib = -0.313151 + 0.000949 * u - 0.000051 * v
        y_calib = 0.821247 - 0.000062 * u - 0.001042 * v
        z_calib = -0.002  # Assume table height for now
        
        return np.array([x_calib, y_calib, z_calib])

    def _compute_hybrid_estimate(self, p_intrinsic, p_empirical, image_center=(320, 240)):
        """
        HYBRID METHOD: Intelligently blend intrinsic and empirical estimates.
        
        Strategy:
        - Use empirical method for points near calibration region (center)
        - Use intrinsic method for edge regions where empirical fitting is less reliable
        - Weighted blend based on distance from image center
        
        Parameters:
            p_intrinsic: [x, y, z] from intrinsics-based method
            p_empirical: [x, y, z] from empirical calibration
            image_center: (u_center, v_center) of calibration region
        
        Returns:
            Blended estimate as numpy array [x, y, z]
        """
        u, v = image_center
        # Empirical method is most accurate near center
        # Weight decreases towards edges (max image size ~640x480)
        distance_from_center = np.sqrt((u - image_center[0])**2 + (v - image_center[1])**2)
        max_distance = np.sqrt(320**2 + 240**2)  # Diagonal to corner
        
        # Weight empirical higher near center, intrinsic higher at edges
        w_empirical = np.exp(-distance_from_center / (max_distance * 0.6))
        w_intrinsic = 1.0 - w_empirical
        
        # Normalize weights
        total_w = w_empirical + w_intrinsic
        w_empirical /= total_w
        w_intrinsic /= total_w
        
        # Blend results
        p_hybrid = w_empirical * p_empirical + w_intrinsic * p_intrinsic
        
        return p_hybrid, w_empirical, w_intrinsic

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
        # HYBRID CALIBRATION: Combine empirical + intrinsic methods
        # ============================================================
        
        # Compute both estimates
        p_intrinsic = self._compute_intrinsic_estimate(u, v, Z)
        p_empirical = self._compute_empirical_estimate(u, v)
        
        # Blend using intelligent weighting
        # Empirical is highly accurate near center, intrinsic more reliable at edges
        image_center = (320, 240)  # Typical camera resolution center
        p_hybrid, w_emp, w_int = self._compute_hybrid_estimate(
            p_intrinsic, p_empirical, image_center=(u, v)
        )
        
        # Use hybrid blend for final estimate
        response.x = float(p_hybrid[0])
        response.y = float(p_hybrid[1])
        response.z = float(p_hybrid[2])
        
        # Debug logging (can be disabled for production)
        if False:  # Set to True for debugging
            self.get_logger().info(
                f"Pixel ({u}, {v}): Intrinsic=[{p_intrinsic[0]:.4f}, {p_intrinsic[1]:.4f}, {p_intrinsic[2]:.4f}] "
                f"Empirical=[{p_empirical[0]:.4f}, {p_empirical[1]:.4f}, {p_empirical[2]:.4f}] "
                f"Hybrid=[{p_hybrid[0]:.4f}, {p_hybrid[1]:.4f}, {p_hybrid[2]:.4f}] "
                f"(w_emp={w_emp:.3f}, w_int={w_int:.3f})"
            )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PixelToRealNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()