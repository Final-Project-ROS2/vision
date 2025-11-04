import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from your_package.srv import PixelToReal

import numpy as np
from cv_bridge import CvBridge

class PixelToRealNode(Node):
    def __init__(self):
        super().__init__('pixel_to_real_node')

        # Subscribers
        self.create_subscription(CameraInfo, '/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # Service
        self.srv = self.create_service(PixelToReal, '/pixel_to_real', self.handle_pixel_to_real)

        # Internal state
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_image = None
        self.bridge = CvBridge()

        # Known transform (camera -> world)
        self.cam_to_world_R = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        self.cam_to_world_t = np.array([0.5, 0.0, 2.0])

        self.get_logger().info("PixelToReal service node initialized.")

    # ------------------- Callbacks -------------------
    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # ------------------- Service handler -------------------
    def handle_pixel_to_real(self, request, response):
        if self.depth_image is None:
            self.get_logger().warn("No depth image received yet.")
            return response
        if None in (self.fx, self.fy, self.cx, self.cy):
            self.get_logger().warn("Camera intrinsics not yet available.")
            return response

        u, v = request.u, request.v
        if v < 0 or v >= self.depth_image.shape[0] or u < 0 or u >= self.depth_image.shape[1]:
            self.get_logger().warn("Pixel out of range.")
            return response

        z_cam = float(self.depth_image[v, u])
        if np.isnan(z_cam) or z_cam <= 0.0:
            self.get_logger().warn("Invalid depth at pixel.")
            return response

        # Pixel -> camera coordinates
        x_cam = (u - self.cx) * z_cam / self.fx
        y_cam = (v - self.cy) * z_cam / self.fy
        p_cam = np.array([x_cam, y_cam, z_cam])

        # Camera -> world coordinates
        p_world = self.cam_to_world_R @ p_cam + self.cam_to_world_t

        response.x = float(p_world[0])
        response.y = float(p_world[1])
        response.z = float(p_world[2])

        self.get_logger().info(f"Pixel ({u},{v}) -> World ({response.x:.3f}, {response.y:.3f}, {response.z:.3f})")

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PixelToRealNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
