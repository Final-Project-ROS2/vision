#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from custom_interfaces.srv import PixelToReal

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import numpy as np
import math


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
        # Translation vector from base_link to camera frame
        # To adjust based on experiments:
        # If pixel_to_real returns (x^, y^, z^) but actual coordinate is (x, y, z):
        #   error = (x - x^, y - y^, z - z^)
        #   t_base_cam_new = t_base_cam_old + error
        # This compensates for calibration errors in the camera position.
        # Practically: put the silver tip of the tape measure where the center of the gripper is, read the distance to the center of the cube
        # see what direction (- or +) the read is, add that
        # Benchmark:
        # LOW: (-0.0386, 0.5303, 0.5238)
        # HIGH: (-0.0361, 0.5303, 0.6458)
        self.t_base_cam = np.array([-0.146, 0.635, 0.8])

        self.R_base_cam = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ])

        # ---- METHOD 2: Empirical linear model ----
        # Least-squares fit on 22 calibration points (5 measures × 2 colours
        # × TR/BL), 2026-03-12.  Model:  x = a0 + a1*u + a2*v
        #                                y = b0 + b1*u + b2*v
        # 2D RMSE = 20.8 mm  (x: 12.1 mm, y: 16.9 mm)
        self._emp_ax = np.array([-0.56859693, +0.00130317, +0.00002114])  # [1,u,v]→x
        self._emp_ay = np.array([+0.98011251, -0.00002728, -0.00133088])  # [1,u,v]→y
        self._emp_z  = 0   # table height (m)

        # ---- Hybrid blend parameters ----
        # Empirical is most accurate near image centre (calibration region).
        # Gaussian decay controls how quickly we trust intrinsics at the edges.
        self._image_center     = (320, 240)   # (u_c, v_c)
        self._blend_sigma_frac = 0.6          # fraction of max diagonal for σ

        self.get_logger().info("PixelToReal with empirical + intrinsic hybrid ready.")

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

    def _intrinsic_estimate(self, u: int, v: int, Z: float) -> np.ndarray:
        """METHOD 1: pinhole back-projection + rigid-body transform to base."""
        ray = self.cam_model.projectPixelTo3dRay((u, v))
        p_cam = np.array(ray) * Z
        p_base = self.R_base_cam @ p_cam + self.t_base_cam
        return p_base

    def _empirical_estimate(self, u: int, v: int) -> np.ndarray:
        """METHOD 2: simple linear model fitted from calibration data.

        Least-squares fit on 22 calibration points, 2026-03-12:
          x = -0.56859693 + 0.00130317*u + 0.00002114*v   (RMS 12.1 mm)
          y =  0.98011251 - 0.00002728*u - 0.00133088*v   (RMS 16.9 mm)
          2D RMSE = 20.8 mm
        """
        feat = np.array([1.0, float(u), float(v)])
        x = float(np.dot(self._emp_ax, feat))
        y = float(np.dot(self._emp_ay, feat))
        return np.array([x, y, self._emp_z])

    def _hybrid_estimate(
        self,
        u: int, v: int,
        p_intrinsic: np.ndarray,
        p_empirical: np.ndarray,
    ) -> tuple:
        """HYBRID: Gaussian-weighted blend.

        Empirical weight is highest at the image centre (calibration region)
        and decays towards the edges; intrinsic picks up the slack.
        """
        uc, vc = self._image_center
        dist  = math.sqrt((u - uc) ** 2 + (v - vc) ** 2)
        sigma = math.sqrt(uc ** 2 + vc ** 2) * self._blend_sigma_frac

        w_emp = math.exp(-dist / sigma)
        w_int = 1.0 - w_emp

        p_hybrid = w_emp * p_empirical + w_int * p_intrinsic
        return p_hybrid, w_emp, w_int

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

        # METHOD 1: intrinsic back-projection
        p_int = self._intrinsic_estimate(u, v, Z)

        # METHOD 2: empirical linear model
        p_emp = self._empirical_estimate(u, v)

        # HYBRID: Gaussian-weighted blend
        p_hybrid, w_emp, w_int = self._hybrid_estimate(u, v, p_int, p_emp)

        response.x = float(p_hybrid[0])
        response.y = float(p_hybrid[1])
        response.z = float(p_hybrid[2])

        self.get_logger().info(
            f'Pixel ({u},{v}) depth={Z:.3f}m | '
            f'Intrinsic=[{p_int[0]:.4f},{p_int[1]:.4f},{p_int[2]:.4f}] '
            f'Empirical=[{p_emp[0]:.4f},{p_emp[1]:.4f}] '
            f'Hybrid=[{p_hybrid[0]:.4f},{p_hybrid[1]:.4f},{p_hybrid[2]:.4f}] '
            f'(w_emp={w_emp:.3f} w_int={w_int:.3f})'
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
