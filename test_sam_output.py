#!/usr/bin/env python3
"""
Test script to verify SAM detector returns JSON with bounding boxes

Usage:
    python3 test_sam_output.py
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import json
import sys


class SAMOutputTester(Node):
    def __init__(self):
        super().__init__('sam_output_tester')
        
        # Create service client
        self.client = self.create_client(Trigger, '/vision/detect_objects')
        
        # Wait for service
        self.get_logger().info("⏳ Waiting for /vision/detect_objects service...")
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("❌ Service not available")
            self.get_logger().error("   Make sure to run: ros2 run vision simple_sam_detector")
            sys.exit(1)
        
        self.get_logger().info("✅ Service found!")
    
    def test_output(self):
        """Call SAM detector and verify output has bounding boxes"""
        self.get_logger().info("=" * 80)
        self.get_logger().info("📞 Calling /vision/detect_objects service...")
        self.get_logger().info("=" * 80)
        
        # Create request
        request = Trigger.Request()
        
        # Call service
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            
            self.get_logger().info("=" * 80)
            self.get_logger().info(f"✅ Service Response Received")
            self.get_logger().info(f"   Success: {response.success}")
            self.get_logger().info("=" * 80)
            
            # Parse JSON
            try:
                data = json.loads(response.message)
                
                self.get_logger().info("📋 JSON Structure:")
                self.get_logger().info(f"   Keys: {list(data.keys())}")
                
                # Check for bounding boxes
                if 'detections' in data and len(data['detections']) > 0:
                    detection_set = data['detections'][0]
                    detections_list = detection_set.get('detections', [])
                    
                    self.get_logger().info(f"   Total detections: {len(detections_list)}")
                    
                    bbox_count = 0
                    for det in detections_list:
                        if 'bbox' in det:
                            bbox_count += 1
                            self.get_logger().info(f"      ✓ Detection {bbox_count}: bbox={det['bbox']}")
                    
                    self.get_logger().info("=" * 80)
                    if bbox_count > 0:
                        self.get_logger().info(f"✅ SUCCESS: Found {bbox_count} bounding boxes in JSON output!")
                    else:
                        self.get_logger().error("❌ FAIL: No bounding boxes found in output!")
                    self.get_logger().info("=" * 80)
                    
                    # Print sample detection
                    if detections_list:
                        self.get_logger().info("📦 Sample Detection JSON:")
                        self.get_logger().info(json.dumps(detections_list[0], indent=2))
                        self.get_logger().info("=" * 80)
                
                else:
                    self.get_logger().error("❌ No detections found in response")
                    
            except json.JSONDecodeError as e:
                self.get_logger().error(f"❌ Failed to parse JSON: {e}")
                self.get_logger().error(f"   Raw response: {response.message[:200]}")
        
        else:
            self.get_logger().error("❌ Service call failed")


def main(args=None):
    rclpy.init(args=args)
    
    tester = SAMOutputTester()
    tester.test_output()
    
    tester.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
