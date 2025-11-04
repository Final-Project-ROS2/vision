#!/usr/bin/env python3
"""
Example: Parse SAM Detection JSON Response

Shows how to call the detection service and parse the JSON schema response.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import json
import sys


class DetectionClient(Node):
    def __init__(self):
        super().__init__('detection_client_example')
        self.client = self.create_client(Trigger, '/vision/detect_objects')
        
    def call_detection(self):
        """Call detection service and parse JSON response"""
        
        # Wait for service
        self.get_logger().info("Waiting for /vision/detect_objects service...")
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service not available!")
            return None
        
        # Create request
        request = Trigger.Request()
        
        # Call service
        self.get_logger().info("Calling detection service...")
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is None:
            self.get_logger().error("Service call failed!")
            return None
        
        response = future.result()
        
        # Parse JSON
        try:
            data = json.loads(response.message)
            return data
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse JSON: {e}")
            return None


def main():
    rclpy.init()
    
    client = DetectionClient()
    
    try:
        # Call detection
        data = client.call_detection()
        
        if data and data.get('success'):
            print("\n" + "="*60)
            print("📊 DETECTION RESULTS")
            print("="*60)
            
            # Print formatted JSON
            print(json.dumps(data, indent=2))
            
            print("\n" + "="*60)
            print("📋 SUMMARY")
            print("="*60)
            
            # Extract key info
            summary = data.get('summary', {})
            print(f"Total Detections: {summary.get('total_detections', 0)}")
            
            if 'average_distance_cm' in summary:
                print(f"Average Distance: {summary['average_distance_cm']} cm")
            
            print(f"Timestamp: {summary.get('timestamp', 'N/A')}")
            
            # Print each detection
            detections = data.get('detections', [])
            if detections:
                print("\n" + "="*60)
                print("🎯 DETECTED OBJECTS")
                print("="*60)
                
                for frame in detections:
                    print(f"\nFrame: {frame.get('image_id', 'N/A')}")
                    
                    for i, det in enumerate(frame.get('detections', []), 1):
                        print(f"\n  Object {i}:")
                        print(f"    Class: {det.get('class_name', 'N/A')}")
                        print(f"    Confidence: {det.get('confidence', 0.0):.2f}")
                        print(f"    BBox: {det.get('bbox', [])}")
                        
                        if 'distance_cm' in det:
                            print(f"    Distance: {det['distance_cm']:.1f} cm")
            
            print("\n" + "="*60)
            print("✅ Parsing complete!")
            print("="*60 + "\n")
            
        else:
            print("\n❌ Detection failed or returned no data\n")
            if data:
                print(json.dumps(data, indent=2))
    
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
