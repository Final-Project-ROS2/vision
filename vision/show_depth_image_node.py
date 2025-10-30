import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthImageViewer(Node):
    def __init__(self):
        super().__init__('depth_image_viewer')

        self.bridge = CvBridge()
        self.latest_depth = None

        # Subscribe to the depth camera topic
        self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        # Service to display the latest depth image
        self.create_service(
            Trigger,
            '/show_depth_image',
            self.show_image_callback
        )

        self.get_logger().info("DepthImageViewer started. Call /show_depth_image to display the latest depth image.")

    def depth_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format (float32)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Clip any invalid depths (NaN or Inf)
            depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            self.latest_depth = depth_image
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def show_image_callback(self, request, response):
        if self.latest_depth is None:
            response.success = False
            response.message = "No depth image received yet."
            return response

        # Normalize depth for visualization
        normalized_depth = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imshow("Depth Camera Image", depth_colormap)
        cv2.waitKey(1)  # Refresh window

        response.success = True
        response.message = "Displayed latest depth image."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DepthImageViewer()

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
