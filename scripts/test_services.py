#!/usr/bin/env python3
"""
Test script for SAM Vision Pipeline Services
Verifies that all services are properly implemented and callable

Usage:
    ros2 run vision test_services
    # or
    python test_services.py
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import time


class ServiceTester(Node):
    """Test node to verify all vision pipeline services"""
    
    def __init__(self):
        super().__init__('service_tester')

        # Define all services to test
        self.service_list = {
            '/vision/detect_objects': 'Object Detection',
            '/vision/classify_objects': 'Object Classification',
            '/vision/generate_grasps': 'Grasp Generation',
            '/vision/get_positions': 'Position Extraction',
            '/vision/build_scene_graph': 'Scene Graph Construction',
            '/vision/process_scene': 'Full Pipeline Processing',
            '/vision/reset_pipeline': 'Pipeline Reset'
        }

        # Create service clients
        self.service_clients = {}
        for service_name in self.service_list.keys():
            self.service_clients[service_name] = self.create_client(Trigger, service_name)
        
        self.get_logger().info("Service Tester Node initialized")
    
    def wait_for_services(self, timeout_sec=10.0):
        """Wait for all services to become available"""
        self.get_logger().info("Waiting for vision pipeline services...")
        
        all_ready = True
        for service_name, description in self.service_list.items():
            self.get_logger().info(f"  Checking {service_name}...")
            
            if not self.service_clients[service_name].wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().error(f"  [FAIL] {service_name} not available")
                all_ready = False
            else:
                self.get_logger().info(f"  [OK] {service_name} is available")
        
        return all_ready
    
    def call_service(self, service_name, description):
        """Call a service and return the result"""
        self.get_logger().info(f"\nCalling {service_name} ({description})...")
        
        try:
            request = Trigger.Request()
            future = self.service_clients[service_name].call_async(request)
            
            # Wait for response
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            
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
    
    def run_service_test_sequence(self):
        """Run a sequential test of all services"""
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("SAM Vision Pipeline Service Test Sequence")
        self.get_logger().info("="*70)
        
        results = {}
        
        # Test 1: Reset Pipeline
        self.get_logger().info("\n[Test 1/7] Testing Pipeline Reset")
        success, msg = self.call_service('/vision/reset_pipeline', 'Pipeline Reset')
        results['reset'] = success
        time.sleep(1)
        
        # Test 2: Object Detection
        self.get_logger().info("\n[Test 2/7] Testing Object Detection")
        success, msg = self.call_service('/vision/detect_objects', 'Object Detection')
        results['detection'] = success
        time.sleep(1)
        
        # Test 3: Object Classification (depends on detection)
        self.get_logger().info("\n[Test 3/7] Testing Object Classification")
        success, msg = self.call_service('/vision/classify_objects', 'Object Classification')
        results['classification'] = success
        time.sleep(1)
        
        # Test 4: Position Extraction (depends on classification)
        self.get_logger().info("\n[Test 4/7] Testing Position Extraction")
        success, msg = self.call_service('/vision/get_positions', 'Position Extraction')
        results['positions'] = success
        time.sleep(1)
        
        # Test 5: Grasp Generation (depends on classification)
        self.get_logger().info("\n[Test 5/7] Testing Grasp Generation")
        success, msg = self.call_service('/vision/generate_grasps', 'Grasp Generation')
        results['grasps'] = success
        time.sleep(1)
        
        # Test 6: Scene Graph (depends on classification and grasps)
        self.get_logger().info("\n[Test 6/7] Testing Scene Graph Construction")
        success, msg = self.call_service('/vision/build_scene_graph', 'Scene Graph')
        results['scene_graph'] = success
        time.sleep(1)
        
        # Test 7: Full Pipeline (all in one)
        self.get_logger().info("\n[Test 7/7] Testing Full Pipeline Processing")
        success, msg = self.call_service('/vision/process_scene', 'Full Pipeline')
        results['full_pipeline'] = success
        
        # Print summary
        self.print_test_summary(results)
    
    def print_test_summary(self, results):
        """Print test results summary"""
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("Test Summary")
        self.get_logger().info("="*70)
        
        total_tests = len(results)
        passed_tests = sum(1 for success in results.values() if success)
        
        for test_name, success in results.items():
            status = "[PASS]" if success else "[FAIL]"
            self.get_logger().info(f"  {status} {test_name}")
        
        self.get_logger().info(f"\nResults: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            self.get_logger().info("All services are working correctly!")
        else:
            self.get_logger().warn(f"{total_tests - passed_tests} test(s) failed")
        
        self.get_logger().info("="*70 + "\n")
    
    def run_individual_service_tests(self):
        """Test each service individually (no dependencies)"""
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("Individual Service Availability Test")
        self.get_logger().info("="*70)
        
        for service_name, description in self.service_list.items():
            self.get_logger().info(f"\nTesting: {service_name}")
            success, msg = self.call_service(service_name, description)
            
            if success:
                self.get_logger().info(f"  Status: OPERATIONAL")
            else:
                self.get_logger().warn(f"  Status: NEEDS ATTENTION")
            
            time.sleep(0.5)
        
        self.get_logger().info("\n" + "="*70)


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        tester = ServiceTester()
        
        # Wait for services
        if not tester.wait_for_services(timeout_sec=15.0):
            tester.get_logger().error("\nSome services are not available!")
            tester.get_logger().info("Make sure the vision pipeline node is running:")
            tester.get_logger().info("  ros2 run vision sam_vision_pipeline")
            tester.get_logger().info("  or")
            tester.get_logger().info("  ros2 launch vision sam_gazebo_complete.launch.py")
            return
        
        tester.get_logger().info("\nAll services are available!")
        time.sleep(1)
        
        # Run sequential test (proper order)
        tester.run_service_test_sequence()
        
        tester.get_logger().info("\nService testing complete!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
