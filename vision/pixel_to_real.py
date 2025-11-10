#!/usr/bin/env python3
"""pixel_to_real_service.py

ROS2 Python node that provides a `/pixel_to_real` service.

`custom_interfaces.srv.PixelToReal`
  - float32 u
  - float32 v
  - string target_frame (optional)
and response fields:
  - float32 x
  - float32 y
  - float32 z
  - bool success
  - string message

This node subscribes to the depth image and camera_info topics, reads the depth
at the requested pixel (with bilinear interpolation), backprojects to camera
coordinates, then transforms the point to the requested `target_frame` using TF2.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import math
import tf2_ros

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
                 info_topic: str = '/camera/color/camera_info',
                 default_target_frame: str = 'world'):
        super().__init__('pixel_to_real_server')
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_rgb_header = None
        self.latest_depth = None
        self.latest_depth_header = None
        self.camera_info = None
        self.default_target_frame = default_target_frame

        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_cb, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_cb, 10)
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_cb, 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create service (deferred type check)
        if PixelToReal is not None:
            self.srv = self.create_service(PixelToReal, 'pixel_to_real', self.handle_pixel_to_real)
            self.get_logger().info('Service /pixel_to_real ready (using custom_interfaces.srv.PixelToReal)')
        else:
            # Create a dummy service using Trigger that returns an error explaining missing srv.
            from std_srvs.srv import Trigger

            def dummy_handle(req, resp):  # type: ignore
                resp.success = False
                resp.message = ('PixelToReal srv type not found. Build your interface package and import ' 
                                '`custom_interfaces.srv.PixelToReal` or adjust this node to use an available srv.')
                return resp

            self.srv = self.create_service(Trigger, 'pixel_to_real', dummy_handle)
            self.get_logger().warning('PixelToReal srv type not importable; created a dummy /pixel_to_real Trigger service that returns an explanatory error.')

    def rgb_cb(self, msg: Image):
        """Store the latest RGB image for pixel coordinate validation."""
        try:
            rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb = rgb_img
            self.latest_rgb_header = msg.header
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")

    def depth_cb(self, msg: Image):
        # Support 32FC1 and 16UC1 encodings; convert to float32 meters.
        try:
            if msg.encoding == '32FC1' or msg.encoding == '32F':
                depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                depth = depth_img.astype(np.float32)
            elif msg.encoding == '16UC1' or msg.encoding == '16U':
                d16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
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

    def read_depth_at(self, u: float, v: float, max_search: int = 5):
        """Read depth with bilinear interpolation; if invalid, search a median window.
        Returns depth in meters or None if unavailable.
        
        Handles invalid depth (NaN/zero/inf) common on reflective surfaces or sensor noise
        by searching a 5-pixel neighborhood and taking the median of valid depth values.
        """
        if self.latest_depth is None:
            return None
        depth = self.latest_depth
        h, w = depth.shape[:2]
        if not (0 <= int(v) < h and 0 <= int(u) < w):
            return None

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

        # Try direct bilinear interpolation first
        d = bilinear(u, v)
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
            self.get_logger().warn(f'No valid depth found within {max_search}px of ({u:.1f},{v:.1f})')
            return None
        
        # Use median to be robust against outliers
        median_depth = float(np.median(valid_depths))
        self.get_logger().info(f'Used median depth {median_depth:.3f}m from {len(valid_depths)} valid neighbors at ({u:.1f},{v:.1f})')
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
        # Support both float and int fields for u/v where available in the custom srv
        try:
            u = float(req.u)
            v = float(req.v)
        except Exception:
            # If the incoming srv is Trigger (dummy), explain error via response
            try:
                resp.success = False
                resp.message = 'Service expects fields u and v (pixel coordinates) in the request.'
                return resp
            except Exception:
                return resp

        target_frame = getattr(req, 'target_frame', self.default_target_frame) or self.default_target_frame

        # Validate pixel coordinates against RGB image dimensions
        if self.latest_rgb is not None:
            rgb_h, rgb_w = self.latest_rgb.shape[:2]
            if not (0 <= int(u) < rgb_w and 0 <= int(v) < rgb_h):
                if hasattr(resp, 'success'):
                    resp.success = False
                if hasattr(resp, 'message'):
                    resp.message = f'Pixel ({u:.1f},{v:.1f}) out of RGB image bounds ({rgb_w}x{rgb_h})'
                return resp

        if self.camera_info is None:
            # Fill response fields if present
            if hasattr(resp, 'success'):
                resp.success = False
            if hasattr(resp, 'message'):
                resp.message = 'No camera_info available yet.'
            return resp

        d = self.read_depth_at(u, v)
        if d is None:
            if hasattr(resp, 'success'):
                resp.success = False
            if hasattr(resp, 'message'):
                resp.message = f'Invalid or missing depth at pixel ({u:.1f},{v:.1f})'
            return resp

        cam_pt = self.backproject(u, v, d)

        # Create PointStamped in camera frame
        ps = PointStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        cam_frame = self.camera_info.header.frame_id if self.camera_info.header.frame_id else 'camera_link'
        ps.header.frame_id = cam_frame
        ps.point.x = float(cam_pt[0])
        ps.point.y = float(cam_pt[1])
        ps.point.z = float(cam_pt[2])

        # Transform to target frame
        try:
            # Use latest available transform
            tf_t = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(target_frame, cam_frame, tf_t)

            # Convert transform to matrix
            q = trans.transform.rotation
            t = trans.transform.translation
            quat = [q.x, q.y, q.z, q.w]

            # Build rotation matrix from quaternion
            # Avoid extra dependency by implementing a small quaternion->matrix converter
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            R = np.array([
                [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
                [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
            ], dtype=np.float64)

            T = np.eye(4, dtype=np.float64)
            T[0:3, 0:3] = R
            T[0:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float64)

            cam_pt_h = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0], dtype=np.float64)
            world_pt_h = T.dot(cam_pt_h)
            x_w, y_w, z_w = world_pt_h[0:3].tolist()

        except Exception as e:
            self.get_logger().error(f'TF transform failed: {e}')
            if hasattr(resp, 'success'):
                resp.success = False
            if hasattr(resp, 'message'):
                resp.message = f'TF transform failed: {e}'
            return resp

        # Fill response
        if hasattr(resp, 'x'):
            resp.x = float(x_w)
        if hasattr(resp, 'y'):
            resp.y = float(y_w)
        if hasattr(resp, 'z'):
            resp.z = float(z_w)
        if hasattr(resp, 'success'):
            resp.success = True
        if hasattr(resp, 'message'):
            resp.message = 'OK'
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
