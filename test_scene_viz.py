#!/usr/bin/env python3
"""
Test scene_understanding visualization directly
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class TestSceneViz(Node):
    def __init__(self):
        super().__init__('test_scene_viz')
        self.bridge = CvBridge()
        self.frame_count = 0
        
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10
        )
        
        cv2.namedWindow('Test Scene Understanding', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Test Scene Understanding', 800, 600)
        
        self.get_logger().info('Test Scene Understanding Visualization Started')
    
    def callback(self, msg):
        self.frame_count += 1
        
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            if self.frame_count == 1:
                self.get_logger().info(f"First image: {img.shape}, dtype: {img.dtype}")
            
            # Add test overlay
            cv2.putText(img, f"Scene Understanding Test | Frame: {self.frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(img, "Waiting for scene analysis...", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('Test Scene Understanding', img)
            cv2.waitKey(1)
            
            if self.frame_count % 30 == 0:
                self.get_logger().info(f"Processed {self.frame_count} frames")
                
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    node = TestSceneViz()
    
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
