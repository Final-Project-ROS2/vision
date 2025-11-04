#!/usr/bin/env python3
"""
Quick viewer for vision pipeline detection results.
Subscribes to /vision/debug_image and displays annotated detections.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class DetectionViewer(Node):
    def __init__(self):
        super().__init__('detection_viewer')
        self.bridge = CvBridge()
        
        # Subscribe to debug image with detections
        self.subscription = self.create_subscription(
            Image,
            '/vision/debug_image',
            self.image_callback,
            10
        )
        
        self.get_logger().info('🔍 Detection Viewer started!')
        self.get_logger().info('   Subscribing to: /vision/debug_image')
        self.get_logger().info('   Press Q to quit')
        
        cv2.namedWindow('Vision Pipeline Detections', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vision Pipeline Detections', 800, 600)
        
    def image_callback(self, msg):
        """Display the detection image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Display
            cv2.imshow('Vision Pipeline Detections', cv_image)
            key = cv2.waitKey(1)
            
            if key == ord('q') or key == ord('Q'):
                self.get_logger().info('Quitting...')
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error displaying image: {e}')


def main(args=None):
    rclpy.init(args=args)
    viewer = DetectionViewer()
    
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
