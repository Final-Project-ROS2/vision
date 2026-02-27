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

        # Subscribers
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

        # Service
        self.service = self.create_service(
            PixelToReal,
            "/pixel_to_real",
            self.handle_pixel_to_real
        )

        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()

        self.depth_image = None
        self.depth_scale = 0.001  # Typical for RealSense (mm → m)
        self.camera_ready = False

        self.get_logger().info("PixelToReal service ready.")

    # ---------------------------
    # Callbacks
    # ---------------------------

    def depth_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")

    def info_callback(self, msg: CameraInfo):
        self.cam_model.fromCameraInfo(msg)
        self.camera_ready = True

    # ---------------------------
    # Service Handler
    # ---------------------------

    def handle_pixel_to_real(self, request, response):

        if self.depth_image is None or not self.camera_ready:
            self.get_logger().warn("Depth or camera info not ready.")
            response.x = 0.0
            response.y = 0.0
            response.z = 0.0
            return response

        u = request.u
        v = request.v

        # Check bounds
        height, width = self.depth_image.shape
        if not (0 <= u < width and 0 <= v < height):
            self.get_logger().warn("Pixel out of bounds.")
            response.x = 0.0
            response.y = 0.0
            response.z = 0.0
            return response

        depth_raw = self.depth_image[v, u]

        if depth_raw == 0:
            self.get_logger().warn("Invalid depth value (0).")
            response.x = 0.0
            response.y = 0.0
            response.z = 0.0
            return response

        # Convert depth to meters
        Z = float(depth_raw) * self.depth_scale

        # Get normalized 3D ray
        ray = self.cam_model.projectPixelTo3dRay((u, v))

        # Multiply ray by depth
        X = ray[0] * Z
        Y = ray[1] * Z

        response.x = float(X)
        response.y = float(Y)
        response.z = float(Z)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PixelToRealNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()