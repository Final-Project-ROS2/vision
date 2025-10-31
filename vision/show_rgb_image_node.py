import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import numpy as np
import cv2


class RGBImageViewer(Node):
    def __init__(self):
        super().__init__('rgb_image_viewer')

        self.bridge = CvBridge()
        self.latest_image = None

        # Subscribe to the RGB camera topic
        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Service to display the latest image
        self.create_service(
            Trigger,
            '/show_rgb_image',
            self.show_image_callback
        )

        numpy_version = np.__version__
        self.get_logger().info(f"Numpy version: {numpy_version}")
        self.get_logger().info("RGBImageViewer started. Call /show_rgb_image to display the latest RGB image.")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV BGR format for display
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def show_image_callback(self, request, response):
        if self.latest_image is None:
            response.success = False
            response.message = "No image received yet."
            return response

        cv2.imshow("RGB Camera Image", self.latest_image)
        cv2.waitKey(1)  # Refresh window

        response.success = True
        response.message = "Displayed latest RGB image."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = RGBImageViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
