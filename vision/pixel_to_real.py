#!/usr/bin/env python3
"""pixel_to_real_service.py

ROS2 Python node that provides a `/pixel_to_real` service.

Service Interface: `custom_interfaces.srv.PixelToReal`
  Request:
    int32 u     # pixel column (x-coordinate, positive to the right)
    int32 v     # pixel row (y-coordinate, positive downward)
  Response:
    float64 x   # world X coordinate (m)
    float64 y   # world Y coordinate (m, positive forward/away from camera)
    float64 z   # world Z coordinate (m, height above table/ground)

Calibration Methods (Hybrid):
  METHOD 1 – Intrinsic:
    Pinhole back-projection using camera intrinsics + known camera pose (R, t).
    t_base_cam is the tunable offset; R_base_cam captures camera orientation.

  METHOD 2 – Empirical:
    Least-squares fit on 22 calibration points (5 measures × 2 colours × TR/BL),
    2026-03-12, camera position [-0.0361, 0.5303, 0.6458], height 0.67 m:
      x = -0.56859693 + 0.00130317*u + 0.00002114*v   (RMS 12.1 mm)
      y =  0.98011251 - 0.00002728*u - 0.00133088*v   (RMS 16.9 mm)
      2D RMSE = 20.8 mm

  HYBRID:
    Gaussian blend – empirical weighted higher near image centre (calibration
    region), intrinsic weighted higher toward edges.

Tuning:
  • t_base_cam  – translate camera origin in base frame (x, y, z)
  • Empirical coefficients – refit from new calibration measurements

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



"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import math
import cv2

from image_geometry import PinholeCameraModel

try:
    from custom_interfaces.srv import PixelToReal
except Exception:
    PixelToReal = None


class PixelToRealServer(Node):

    def __init__(self):
        super().__init__('pixel_to_real_server')

        self.declare_parameter('real_hardware', True)
        self.real_hardware = bool(self.get_parameter('real_hardware').value)

        if self.real_hardware:
            depth_topic      = '/camera/depth/image_rect_raw'
            info_topic       = '/camera/color/camera_info'
            rgb_topic        = '/camera/color/image_raw'
        else:
            depth_topic      = '/camera/depth/image_raw'
            info_topic       = '/camera/camera_info'
            rgb_topic        = '/camera/image_raw'

        self.bridge     = CvBridge()
        self.cam_model  = PinholeCameraModel()

        self.latest_rgb   = None
        self.latest_depth = None
        self.camera_ready = False

        self.depth_sub = self.create_subscription(Image,      depth_topic, self.depth_cb, 10)
        self.info_sub  = self.create_subscription(CameraInfo, info_topic,  self.info_cb,  10)
        self.rgb_sub   = self.create_subscription(Image,      rgb_topic,   self.rgb_cb,   10)

        self.debug_pub = self.create_publisher(Image, '/pixel_to_real/debug_image', 10)

        # ── Depth scale (16UC1 → metres) ─────────────────────────────────────
        self.depth_scale = 0.001   # mm → m

        # ── METHOD 1: Camera pose in base_link ───────────────────────────────
        # t_base_cam  : camera origin expressed in base frame  ← TUNE THIS
        # R_base_cam  : rotation that maps camera axes to base axes
        #               (camera Z forward → base -Z, camera Y down → base -Y)
        self.t_base_cam = np.array([-0.146, 0.635, 0.8])   # <-- adjust here

        self.R_base_cam = np.array([
            [ 1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ])

        # ── METHOD 2: Empirical linear model ─────────────────────────────────
        # Least-squares fit on 22 calibration points (5 measures × 2 colours
        # × TR/BL), 2026-03-12.  Model:  x = a0 + a1*u + a2*v
        #                                y = b0 + b1*u + b2*v
        # 2D RMSE = 20.8 mm  (x: 12.1 mm, y: 16.9 mm)
        self._emp_ax = np.array([-0.56859693, +0.00130317, +0.00002114])  # [1,u,v]→x
        self._emp_ay = np.array([+0.98011251, -0.00002728, -0.00133088])  # [1,u,v]→y
        self._emp_z  = -0.002   # table height assumption (m)

        # ── Hybrid blend parameters ───────────────────────────────────────────
        # Empirical is most accurate near image centre (calibration region).
        # Gaussian decay controls how quickly we trust intrinsics at the edges.
        self._image_center    = (320, 240)   # (u_c, v_c)
        self._blend_sigma_frac = 0.6         # fraction of max diagonal for σ

        if PixelToReal is not None:
            self.srv = self.create_service(
                PixelToReal, '/pixel_to_real', self.handle_pixel_to_real
            )
            self.get_logger().info('Service /pixel_to_real ready')
        else:
            self.get_logger().error(
                'PixelToReal srv not found – build custom_interfaces first.'
            )
            raise RuntimeError('custom_interfaces.srv.PixelToReal not available')

        self.get_logger().info(
            f't_base_cam = {self.t_base_cam.tolist()}   '
            f'real_hardware={self.real_hardware}'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def rgb_cb(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB convert error: {e}')

    def depth_cb(self, msg: Image):
        try:
            if msg.encoding in ('16UC1', '16U'):
                raw = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                self.latest_depth = raw.astype(np.float32) * self.depth_scale
            elif msg.encoding in ('32FC1', '32F'):
                self.latest_depth = self.bridge.imgmsg_to_cv2(
                    msg, 'passthrough'
                ).astype(np.float32)
            else:
                raw = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                self.latest_depth = raw.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f'Depth convert error: {e}')

    def info_cb(self, msg: CameraInfo):
        self.cam_model.fromCameraInfo(msg)
        self.camera_ready = True

    # ── Depth reading ──────────────────────────────────────────────────────────

    def get_robust_depth(self, u: int, v: int, window: int = 5) -> float | None:
        """Median depth over a small window; returns None if no valid pixels."""
        if self.latest_depth is None:
            return None
        h, w = self.latest_depth.shape
        half = window // 2
        u0, u1 = max(u - half, 0), min(u + half + 1, w)
        v0, v1 = max(v - half, 0), min(v + half + 1, h)
        patch = self.latest_depth[v0:v1, u0:u1]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        return float(np.median(valid)) if len(valid) > 0 else None

    # ── Conversion methods ─────────────────────────────────────────────────────

    def _intrinsic_estimate(self, u: int, v: int, Z: float) -> np.ndarray:
        """METHOD 1: pinhole back-projection + rigid-body transform to base.

        ray = cam_model.projectPixelTo3dRay((u, v))  [unit vector in cam frame]
        p_cam = ray * Z                               [scale to depth]
        p_base = R @ p_cam + t                        [transform to base frame]
        """
        if not self.camera_ready:
            # Fallback: hardcoded D435i intrinsics
            fx, fy, cx, cy = 615.0, 615.0, 320.0, 240.0
            ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        else:
            ray = np.array(self.cam_model.projectPixelTo3dRay((u, v)))

        p_cam  = ray * Z
        p_base = self.R_base_cam @ p_cam + self.t_base_cam
        return p_base

    def _empirical_estimate(self, u: int, v: int) -> np.ndarray:
        """METHOD 2: simple linear model fitted from calibration data."""
        feat = np.array([1.0, float(u), float(v)])
        x = float(np.dot(self._emp_ax, feat))
        y = float(np.dot(self._emp_ay, feat))
        return np.array([x, y, self._emp_z])

    def _hybrid_estimate(
        self,
        u: int, v: int,
        p_intrinsic: np.ndarray,
        p_empirical: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """HYBRID: Gaussian-weighted blend.

        Empirical weight is highest at the image centre (calibration region)
        and decays towards the edges; intrinsic picks up the slack.
        """
        uc, vc = self._image_center
        dist   = math.sqrt((u - uc) ** 2 + (v - vc) ** 2)
        sigma  = math.sqrt(uc ** 2 + vc ** 2) * self._blend_sigma_frac  # ~σ in pixels

        w_emp  = math.exp(-dist / sigma)
        w_int  = 1.0 - w_emp

        p_hybrid = w_emp * p_empirical + w_int * p_intrinsic
        return p_hybrid, w_emp, w_int

    # ── Service handler ────────────────────────────────────────────────────────

    def handle_pixel_to_real(self, request, response):
        u = int(request.u)
        v = int(request.v)

        # ── Depth ─────────────────────────────────────────────────────────────
        Z = self.get_robust_depth(u, v)
        if Z is None:
            self.get_logger().warn(f'No valid depth at ({u},{v}), using 0.8 m fallback.')
            Z = 0.8   # fallback to estimated table depth

        # ── Compute both estimates ────────────────────────────────────────────
        p_int = self._intrinsic_estimate(u, v, Z)
        p_emp = self._empirical_estimate(u, v)

        # ── Blend ─────────────────────────────────────────────────────────────
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

        # ── Debug image ───────────────────────────────────────────────────────
        self._publish_debug(u, v, p_hybrid, Z)

        return response

    def _publish_debug(self, u: int, v: int, p: np.ndarray, depth: float):
        if self.latest_rgb is None:
            return
        img = self.latest_rgb.copy()
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.circle(img, (u, v), 20, (0, 255, 0), 2)
        cv2.putText(img, f'Pixel: ({u},{v})',
                    (u + 25, v - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f'World: ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})m',
                    (u + 25, v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(img, f'Depth: {depth:.3f}m',
                    (u + 25, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        uc, vc = self._image_center
        cv2.drawMarker(img, (uc, vc), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 30, 2)
        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Debug publish error: {e}')


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
