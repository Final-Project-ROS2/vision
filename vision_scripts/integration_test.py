#!/usr/bin/env python3
"""
Integration Test for SAM Vision Pipeline
Tests all services with RGB image from Final-proj/src/arrange.jpg

This script:
1. Publishes test image to camera topics
2. Calls all vision services in correct order
3. Validates responses
4. Reports results

Usage:
    # Terminal 1: Start pipeline
    ros2 run vision sam_vision_pipeline
    
    # Terminal 2: Run this test
    python scripts/integration_test.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path
import time


class IntegrationTester(Node):
    """Integration test node for SAM vision pipeline"""
    
    def __init__(self):
        super().__init__('integration_tester')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        # Service clients
        self.services = {
            'reset': self.create_client(Trigger, '/vision/reset_pipeline'),
            'detect': self.create_client(Trigger, '/vision/detect_objects'),
            'classify': self.create_client(Trigger, '/vision/classify_objects'),
            'positions': self.create_client(Trigger, '/vision/get_positions'),
            'grasps': self.create_client(Trigger, '/vision/generate_grasps'),
            'scene_graph': self.create_client(Trigger, '/vision/build_scene_graph'),
        }
        
        # Load test image
        self.test_image = None
        self.test_depth = None
        self.load_test_images()
        
        # Publishing timer
        self.timer = self.create_timer(0.5, self.publish_images)
        
        self.get_logger().info("="*70)
        self.get_logger().info("SAM Vision Pipeline Integration Test")
        self.get_logger().info("="*70)
    
    def load_test_images(self):
        """Load test image from src"""
        image_path = Path("Final-proj/src/arrange.jpg")
        
        if not image_path.exists():
            self.get_logger().error(f"Test image not found: {image_path}")
            return
        
        self.test_image = cv2.imread(str(image_path))
        if self.test_image is not None:
            self.get_logger().info(f"Loaded test image: {image_path}")
            self.get_logger().info(f"Image shape: {self.test_image.shape}")
            
            # Create synthetic depth
            gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
            self.test_depth = (255 - gray).astype(np.uint16) * 10
            self.get_logger().info("Created synthetic depth image")
        else:
            self.get_logger().error(f"Failed to load image: {image_path}")
    
    def publish_images(self):
        """Publish test images to camera topics"""
        if self.test_image is None:
            return
        
        # Publish RGB
        try:
            rgb_msg = self.bridge.cv2_to_imgmsg(self.test_image, encoding='bgr8')
            rgb_msg.header.stamp = self.get_clock().now().to_msg()
            rgb_msg.header.frame_id = 'camera_link'
            self.rgb_pub.publish(rgb_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish RGB: {e}")
        
        # Publish depth
        if self.test_depth is not None:
            try:
                depth_msg = self.bridge.cv2_to_imgmsg(self.test_depth, encoding='passthrough')
                depth_msg.header.stamp = self.get_clock().now().to_msg()
                depth_msg.header.frame_id = 'camera_link'
                self.depth_pub.publish(depth_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish depth: {e}")
        
        # Publish camera info
        cam_info = CameraInfo()
        cam_info.header.stamp = self.get_clock().now().to_msg()
        cam_info.header.frame_id = 'camera_link'
        cam_info.height = self.test_image.shape[0]
        cam_info.width = self.test_image.shape[1]
        self.camera_info_pub.publish(cam_info)
    
    def wait_for_services(self, timeout_sec=10.0):
        """Wait for all services"""
        self.get_logger().info("\nWaiting for services...")
        
        all_ready = True
        for name, client in self.services.items():
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().error(f"  [FAIL] Service not available: /vision/{name}")
                all_ready = False
            else:
                self.get_logger().info(f"  [OK] Service ready: /vision/{name}")
        
        return all_ready
    
    def call_service(self, name, client):
        """Call a service and return result"""
        self.get_logger().info(f"\nCalling /vision/{name}...")
        
        try:
            request = Trigger.Request()
            future = client.call_async(request)
            
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            
            if future.result() is not None:
                response = future.result()
                if response.success:
                    self.get_logger().info(f"  [SUCCESS] {response.message}")
                    return True, response.message
                else:
                    self.get_logger().warn(f"  [FAILED] {response.message}")
                    return False, response.message
            else:
                self.get_logger().error(f"  [TIMEOUT] Service call timed out")
                return False, "Timeout"
        except Exception as e:
            self.get_logger().error(f"  [ERROR] {e}")
            return False, str(e)
    
    def run_integration_test(self):
        """Run full integration test"""
        if self.test_image is None:
            self.get_logger().error("No test image loaded. Aborting test.")
            return
        
        # Wait for services
        if not self.wait_for_services():
            self.get_logger().error("\nNot all services available!")
            self.get_logger().info("Make sure the pipeline is running:")
            self.get_logger().info("  ros2 run vision sam_vision_pipeline")
            return
        
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("Starting Integration Test")
        self.get_logger().info("="*70)
        
        # Give time for images to be published
        self.get_logger().info("\nWaiting for images to be published...")
        time.sleep(2)
        
        # Test sequence
        results = {}
        
        # Test 1: Reset
        success, msg = self.call_service('reset', self.services['reset'])
        results['Reset Pipeline'] = success
        time.sleep(1)
        
        # Test 2: Detection
        success, msg = self.call_service('detect', self.services['detect'])
        results['Object Detection'] = success
        time.sleep(1)
        
        # Test 3: Classification
        success, msg = self.call_service('classify', self.services['classify'])
        results['Classification'] = success
        time.sleep(1)
        
        # Test 4: Positions
        success, msg = self.call_service('positions', self.services['positions'])
        results['Position Extraction'] = success
        time.sleep(1)
        
        # Test 5: Grasps
        success, msg = self.call_service('grasps', self.services['grasps'])
        results['Grasp Generation'] = success
        time.sleep(1)
        
        # Test 6: Scene Graph
        success, msg = self.call_service('scene_graph', self.services['scene_graph'])
        results['Scene Graph'] = success
        
        # Print summary
        self.print_summary(results)
        
        # Stop publishing
        self.timer.cancel()
    
    def print_summary(self, results):
        """Print test summary"""
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("Integration Test Summary")
        self.get_logger().info("="*70)
        
        passed = sum(1 for s in results.values() if s)
        total = len(results)
        
        for test_name, success in results.items():
            status = "[PASS]" if success else "[FAIL]"
            self.get_logger().info(f"  {status} {test_name}")
        
        self.get_logger().info(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            self.get_logger().info("\n✓ All tests PASSED!")
        else:
            self.get_logger().warn(f"\n✗ {total - passed} test(s) FAILED")
        
        self.get_logger().info("="*70)


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        tester = IntegrationTester()
        
        # Wait a bit for initialization
        time.sleep(1)
        
        # Run test
        tester.run_integration_test()
        
        # Keep node alive briefly
        time.sleep(2)
        
        tester.destroy_node()
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
