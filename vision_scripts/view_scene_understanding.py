#!/usr/bin/env python3
"""
Visualization Subscriber for Scene Understanding

This script subscribes to the /vision/scene_understanding_viz topic and
displays the received image in an OpenCV window.

Subscribes to:
    /vision/scene_understanding_viz (sensor_msgs/msg/Image)

How to Run:
    1. Make sure the scene_understanding_node is running and publishing
       the visualization image.
    
    2. In a new terminal, run this script:
       ros2 run vision view_scene_understanding
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VizSubscriber(Node):
    def __init__(self):
        super().__init__('viz_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/vision/scene_understanding_viz',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.window_name = "Scene Understanding Visualization"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def image_callback(self, msg):
        self.get_logger().info("Received visualization image.")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow(self.window_name, cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to display image: {e}")

def main(args=None):
    rclpy.init(args=args)
    viz_subscriber = VizSubscriber()
    rclpy.spin(viz_subscriber)
    viz_subscriber.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
