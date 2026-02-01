#!/usr/bin/env python3
"""
Debug script to test image conversion from /camera/image_raw
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DebugImageTest(Node):
    def __init__(self):
        super().__init__('debug_image_test')
        self.bridge = CvBridge()
        self.frame_count = 0
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        cv2.namedWindow('Debug Image Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug Image Test', 640, 480)
        
        self.get_logger().info('Debug Image Test Started')
        self.get_logger().info('Subscribing to /camera/image_raw')
    
    def image_callback(self, msg):
        self.frame_count += 1
        
        try:
            # Log message details
            if self.frame_count == 1:
                self.get_logger().info(f"First image received!")
                self.get_logger().info(f"  Encoding: {msg.encoding}")
                self.get_logger().info(f"  Size: {msg.width}x{msg.height}")
                self.get_logger().info(f"  Step: {msg.step}")
                self.get_logger().info(f"  Data length: {len(msg.data)}")
            
            # Try conversion to BGR8 (what graspnet uses)
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            if self.frame_count == 1:
                self.get_logger().info(f"  Converted shape: {img_bgr.shape}")
                self.get_logger().info(f"  Converted dtype: {img_bgr.dtype}")
                self.get_logger().info(f"  Min/Max values: {img_bgr.min()}/{img_bgr.max()}")
            
            # Add frame counter text
            cv2.putText(img_bgr, f"Frame: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            cv2.putText(img_bgr, f"Encoding: {msg.encoding}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show image
            cv2.imshow('Debug Image Test', img_bgr)
            cv2.waitKey(1)
            
            if self.frame_count % 30 == 0:
                self.get_logger().info(f"Processed {self.frame_count} frames successfully")
                
        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DebugImageTest()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
