#!/usr/bin/env python3
"""
Test Subscriber for SAM Detections

Subscribe to /vision/sam_detections topic and print received messages.
Use this to verify that simple_sam_detector is publishing correctly.

Usage:
    python3 test_sam_subscriber.py
    
Then in another terminal:
    ros2 service call /vision/detect_objects std_srvs/srv/Trigger
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # Placeholder - will be SAMDetections after build


class SAMDetectionSubscriber(Node):
    """Test subscriber to verify SAM detection publishing"""
    
    def __init__(self):
        super().__init__('sam_detection_subscriber')
        
        # Subscribe to SAM detections topic
        self.subscription = self.create_subscription(
            Image,  # Placeholder type - replace with SAMDetections after build
            '/vision/sam_detections',
            self.detection_callback,
            10
        )
        
        self.message_count = 0
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("SAM Detection Subscriber Started")
        self.get_logger().info("=" * 80)
        self.get_logger().info("Listening to: /vision/sam_detections")
        self.get_logger().info("Waiting for detection messages...")
        self.get_logger().info("=" * 80)
        self.get_logger().info("Trigger detection with:")
        self.get_logger().info("  ros2 service call /vision/detect_objects std_srvs/srv/Trigger")
        self.get_logger().info("=" * 80)
    
    def detection_callback(self, msg):
        """Handle incoming detection messages"""
        self.message_count += 1
        
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"RECEIVED MESSAGE #{self.message_count}")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Message Type: {type(msg).__name__}")
        self.get_logger().info(f"Encoding: {msg.encoding if hasattr(msg, 'encoding') else 'N/A'}")
        self.get_logger().info(f"Message received successfully!")
        self.get_logger().info("=" * 80)
        
        # After building with SAMDetections message, you can access:
        # self.get_logger().info(f"Frame ID: {msg.header.frame_id}")
        # self.get_logger().info(f"Image ID: {msg.image_id}")
        # self.get_logger().info(f"Total Detections: {msg.total_detections}")
        # self.get_logger().info(f"Average Distance: {msg.average_distance_cm} cm")
        # 
        # for i, det in enumerate(msg.detections):
        #     self.get_logger().info(f"  [{i}] {det.class_name}: confidence={det.confidence:.2f}")
        #     self.get_logger().info(f"      bbox={det.bbox}, center={det.center}")
        #     self.get_logger().info(f"      area={det.area}, distance={det.distance_cm} cm")


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = SAMDetectionSubscriber()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
