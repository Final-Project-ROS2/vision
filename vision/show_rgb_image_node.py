#!/usr/bin/env python3
# Copyright 2025 final-project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2


class RGBImageViewer(Node):
    def __init__(self):
        super().__init__('rgb_image_viewer')

        self.bridge = CvBridge()
        self.latest_image = None
        self.show_continuous = False

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

        # Service to toggle continuous display
        self.create_service(
            Trigger,
            '/toggle_continuous_display',
            self.toggle_continuous_callback
        )

        self.get_logger().info("RGBImageViewer started.")
        self.get_logger().info("Services: /show_rgb_image, /toggle_continuous_display")
        self.get_logger().info("Subscribing to: /camera/image_raw")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV BGR format for display using CvBridge
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Show image continuously if enabled
            if self.show_continuous:
                cv2.imshow("RGB Camera Feed (Continuous)", self.latest_image)
                cv2.waitKey(1)
                
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

    def toggle_continuous_callback(self, request, response):
        self.show_continuous = not self.show_continuous
        if self.show_continuous:
            response.success = True
            response.message = "Continuous display ON"
            self.get_logger().info("Continuous display enabled")
        else:
            response.success = True
            response.message = "Continuous display OFF"
            cv2.destroyWindow("RGB Camera Feed (Continuous)")
            self.get_logger().info("Continuous display disabled")
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
