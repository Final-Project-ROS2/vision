#!/usr/bin/env python3
"""
ROS2 Vision Pipeline Test Node
Tests the vision pipeline with RGB and RGB-D images from src/pipeline

This node can:
1. Load test images from disk (RGB only or RGB-D)
2. Publish them to camera topics
3. Call vision services to process them
4. Validate the results

Usage:
    # Test with RGB only
    ros2 run vision test_pipeline_images --ros-args -p image_path:=/path/to/image.jpg
    
    # Test with RGB-D
    ros2 run vision test_pipeline_images --ros-args -p image_path:=/path/to/rgb.jpg -p depth_path:=/path/to/depth.png
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
import sys
import os


class PipelineImageTester(Node):
    """Test node for vision pipeline with test images"""
    
    def __init__(self):
        super().__init__('pipeline_image_tester')
        
        # Parameters
        self.declare_parameter('image_path', '')
        self.declare_parameter('depth_path', '')
        self.declare_parameter('test_mode', 'rgb')  # 'rgb' or 'rgbd'
        self.declare_parameter('auto_test', True)
        self.declare_parameter('publish_rate', 1.0)  # Hz
        
        # Get parameters
        self.image_path = self.get_parameter('image_path').value
        self.depth_path = self.get_parameter('depth_path').value
        self.test_mode = self.get_parameter('test_mode').value
        self.auto_test = self.get_parameter('auto_test').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        # Service clients
        self.detect_client = self.create_client(Trigger, '/vision/detect_objects')
        self.classify_client = self.create_client(Trigger, '/vision/classify_objects')
        self.grasp_client = self.create_client(Trigger, '/vision/generate_grasps')
        self.position_client = self.create_client(Trigger, '/vision/get_positions')
        self.scene_graph_client = self.create_client(Trigger, '/vision/build_scene_graph')
        self.process_client = self.create_client(Trigger, '/vision/process_scene')
        
        # Load test images
        self.rgb_image = None
        self.depth_image = None
        self.load_test_images()
        
        # Publishing timer
        publish_rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_images)
        
        self.get_logger().info("="*70)
        self.get_logger().info("Vision Pipeline Image Tester initialized")
        self.get_logger().info(f"Test mode: {self.test_mode.upper()}")
        self.get_logger().info(f"RGB image: {self.image_path if self.image_path else 'Not provided'}")
        self.get_logger().info(f"Depth image: {self.depth_path if self.depth_path else 'Not provided'}")
        self.get_logger().info("="*70)
        
        # Auto test if enabled
        if self.auto_test and self.rgb_image is not None:
            self.create_timer(3.0, self.run_auto_test)  # Run after 3 seconds
    
    def load_test_images(self):
        """Load test images from disk"""
        # Find test images if not specified
        if not self.image_path:
            self.image_path = self.find_test_image()
        
        # Load RGB image
        if self.image_path and Path(self.image_path).exists():
            self.rgb_image = cv2.imread(str(self.image_path))
            if self.rgb_image is not None:
                self.get_logger().info(f"Loaded RGB image: {self.image_path}")
                self.get_logger().info(f"  Shape: {self.rgb_image.shape}")
            else:
                self.get_logger().error(f"Failed to load RGB image: {self.image_path}")
        else:
            self.get_logger().warn(f"RGB image not found: {self.image_path}")
        
        # Load depth image if provided
        if self.depth_path and Path(self.depth_path).exists():
            self.depth_image = cv2.imread(str(self.depth_path), cv2.IMREAD_UNCHANGED)
            if self.depth_image is not None:
                self.get_logger().info(f"Loaded depth image: {self.depth_path}")
                self.get_logger().info(f"  Shape: {self.depth_image.shape}")
                self.test_mode = 'rgbd'
            else:
                self.get_logger().error(f"Failed to load depth image: {self.depth_path}")
        elif self.test_mode == 'rgbd':
            # Create synthetic depth if in RGBD mode but no depth provided
            if self.rgb_image is not None:
                self.depth_image = self.create_synthetic_depth(self.rgb_image)
                self.get_logger().info("Created synthetic depth image")
    
    def find_test_image(self):
        """Find a test image in the workspace"""
        possible_paths = [
            "Final-proj/data/test_images/arrange.jpg",
            "Final-proj/src/arrange.jpg",
            "data/test_images/sample.jpg",
            "../test_image.jpg",
            "test_image.jpg"
        ]
        
        for path in possible_paths:
            full_path = Path(path)
            if full_path.exists():
                return str(full_path)
        
        self.get_logger().warn("No test image found in common locations")
        return None
    
    def create_synthetic_depth(self, rgb_image):
        """Create synthetic depth image from RGB"""
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        # Simple depth: darker = farther
        depth = (255 - gray).astype(np.uint16) * 10
        return depth
    
    def publish_images(self):
        """Publish test images to camera topics"""
        if self.rgb_image is None:
            return
        
        # Publish RGB
        try:
            rgb_msg = self.bridge.cv2_to_imgmsg(self.rgb_image, encoding='bgr8')
            rgb_msg.header.stamp = self.get_clock().now().to_msg()
            rgb_msg.header.frame_id = 'camera_link'
            self.rgb_pub.publish(rgb_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish RGB: {e}")
        
        # Publish depth if available
        if self.depth_image is not None:
            try:
                depth_msg = self.bridge.cv2_to_imgmsg(self.depth_image, encoding='passthrough')
                depth_msg.header.stamp = self.get_clock().now().to_msg()
                depth_msg.header.frame_id = 'camera_link'
                self.depth_pub.publish(depth_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish depth: {e}")
        
        # Publish camera info
        cam_info = CameraInfo()
        cam_info.header.stamp = self.get_clock().now().to_msg()
        cam_info.header.frame_id = 'camera_link'
        cam_info.height = self.rgb_image.shape[0]
        cam_info.width = self.rgb_image.shape[1]
        self.camera_info_pub.publish(cam_info)
    
    def run_auto_test(self):
        """Run automated test sequence"""
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("Starting Automated Test Sequence")
        self.get_logger().info("="*70)
        
        # Wait for services
        self.get_logger().info("\nWaiting for vision services...")
        if not self.wait_for_all_services(timeout_sec=10.0):
            self.get_logger().error("Not all services available. Aborting test.")
            return
        
        self.get_logger().info("All services available!\n")
        time.sleep(1)
        
        # Test sequence
        tests = [
            ('Detection', self.detect_client, '/vision/detect_objects'),
            ('Classification', self.classify_client, '/vision/classify_objects'),
            ('Position Extraction', self.position_client, '/vision/get_positions'),
        ]
        
        # Add grasp test only if depth available
        if self.depth_image is not None:
            tests.append(('Grasp Generation', self.grasp_client, '/vision/generate_grasps'))
            tests.append(('Scene Graph', self.scene_graph_client, '/vision/build_scene_graph'))
        
        # Run tests
        results = {}
        for name, client, service_name in tests:
            self.get_logger().info(f"Testing: {name}")
            success, message = self.call_service_sync(client, service_name)
            results[name] = success
            
            if success:
                self.get_logger().info(f"  [PASS] {message}")
            else:
                self.get_logger().warn(f"  [FAIL] {message}")
            
            time.sleep(1)
        
        # Print summary
        self.print_test_summary(results)
        
        # Cancel timer so it only runs once
        self.timer.cancel()
    
    def wait_for_all_services(self, timeout_sec=10.0):
        """Wait for all services to become available"""
        services = [
            self.detect_client,
            self.classify_client,
            self.position_client,
            self.grasp_client,
            self.scene_graph_client,
            self.process_client
        ]
        
        all_ready = True
        for client in services:
            if not client.wait_for_service(timeout_sec=timeout_sec):
                all_ready = False
        
        return all_ready
    
    def call_service_sync(self, client, service_name):
        """Call service synchronously and return result"""
        try:
            request = Trigger.Request()
            future = client.call_async(request)
            
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            
            if future.result() is not None:
                response = future.result()
                return response.success, response.message
            else:
                return False, "Timeout"
        except Exception as e:
            return False, str(e)
    
    def print_test_summary(self, results):
        """Print test results summary"""
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("Test Summary")
        self.get_logger().info("="*70)
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        for test_name, success in results.items():
            status = "[PASS]" if success else "[FAIL]"
            self.get_logger().info(f"  {status} {test_name}")
        
        self.get_logger().info(f"\nResults: {passed}/{total} tests passed")
        self.get_logger().info("="*70 + "\n")


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        tester = PipelineImageTester()
        rclpy.spin(tester)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
