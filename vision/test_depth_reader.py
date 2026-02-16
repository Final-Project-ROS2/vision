#!/usr/bin/env python3
"""
Standalone script to test reading depth values from RealSense camera.
Opens depth camera via ROS2 topic and displays depth at clicked pixel.

Usage:
    cd ~/final_project_ws && source ./vision_venv/bin/activate
    cd src/vision && python3 vision/test_depth_reader.py
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DepthReader(Node):
    """Simple depth camera reader for testing."""
    
    def __init__(self):
        super().__init__('depth_reader')
        
        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_rgb = None
        
        # Subscribe to depth and RGB topics
        self.depth_sub = self.create_subscription(
            Image, 
            '/camera/depth/image_raw', 
            self.depth_callback, 
            10
        )
        self.rgb_sub = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.rgb_callback, 
            10
        )
        
        self.get_logger().info('Depth reader started. Click on image to read depth.')
        
    def rgb_callback(self, msg):
        """Store latest RGB image."""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB error: {e}')
    
    def depth_callback(self, msg):
        """Store latest depth image."""
        try:
            if msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                self.latest_depth = depth.astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                if depth.dtype == np.uint16:
                    self.latest_depth = depth.astype(np.float32) / 1000.0
                else:
                    self.latest_depth = depth
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')
    
    def get_depth_at_pixel(self, u, v):
        """Read depth value at pixel (u, v)."""
        if self.latest_depth is None:
            return None
        
        h, w = self.latest_depth.shape
        if not (0 <= u < w and 0 <= v < h):
            return None
        
        return float(self.latest_depth[v, u])
    
    def show_depth_image(self):
        """Display depth image with interactive clicking."""
        if self.latest_depth is None or self.latest_rgb is None:
            return
        
        # Create visualization
        display = self.latest_rgb.copy()
        
        # Normalize depth for visualization
        depth_vis = self.latest_depth.copy()
        depth_vis = np.nan_to_num(depth_vis, 0)
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-6)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        # Combine RGB and depth
        combined = np.hstack([display, depth_colored])
        
        cv2.imshow('RGB | Depth (click to read)', combined)
        cv2.setMouseCallback('RGB | Depth (click to read)', self.mouse_callback)
        cv2.waitKey(1)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to read depth."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check which side was clicked
            if self.latest_rgb is not None:
                w = self.latest_rgb.shape[1]
                if x < w:  # Clicked on RGB side
                    u, v = x, y
                else:  # Clicked on depth side
                    u, v = x - w, y
                
                depth = self.get_depth_at_pixel(u, v)
                if depth is not None:
                    print(f"\nPixel ({u}, {v}): depth = {depth:.4f}m")
                    self.get_logger().info(f'Pixel ({u}, {v}): depth = {depth:.4f}m')


def main():
    rclpy.init()
    
    node = DepthReader()
    
    print("\n" + "="*60)
    print("DEPTH READER - Click on image to read depth values")
    print("="*60)
    print("Press 'q' to quit\n")
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.show_depth_image()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
