#!/usr/bin/env python3
"""
ROS2 Camera Info Publisher Node

This node continuously publishes camera intrinsic parameters to the 
/camera/depth/camera_info topic for use by other vision nodes.

Topic: /camera/depth/camera_info
Type: sensor_msgs/msg/CameraInfo
Rate: 30 Hz

Usage:
    ros2 run vision real_cam_info
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header


class CameraInfoPublisher(Node):
    """ROS2 node that publishes camera intrinsic parameters."""
    
    def __init__(self):
        super().__init__('real_cam_info_node')
        
        # Create publisher
        self.publisher_ = self.create_publisher(
            CameraInfo,
            '/camera/depth/camera_info',
            10
        )
        
        # Publish at 30 Hz
        self.timer_period = 1.0 / 30.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.get_logger().info('Camera info publisher started')
        self.get_logger().info('Publishing to: /camera/depth/camera_info at 30 Hz')
    
    def timer_callback(self):
        """Publish camera info message."""
        msg = CameraInfo()
        
        # Header with current timestamp
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_optical_link'
        
        # Image dimensions
        msg.height = 480
        msg.width = 640
        
        # Distortion model and parameters
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Camera intrinsic matrix K (3x3)
        msg.k = [
            390.50704956, 0.0, 322.57781982,
            0.0, 390.50704956, 235.13317871,
            0.0, 0.0, 1.0
        ]
        
        # Rectification matrix R (3x3)
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        
        # Projection matrix P (3x4)
        msg.p = [
            528.433756558705, 0.0, 320.5, -0.0,
            0.0, 528.433756558705, 240.5, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        
        # Binning
        msg.binning_x = 0
        msg.binning_y = 0
        
        # Region of Interest (ROI)
        msg.roi.x_offset = 0
        msg.roi.y_offset = 0
        msg.roi.height = 0
        msg.roi.width = 0
        msg.roi.do_rectify = False
        
        # Publish
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CameraInfoPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
