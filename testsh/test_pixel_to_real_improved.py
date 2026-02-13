#!/usr/bin/env python3
"""
Test script for pixel_to_real_world service with coordinate validation.

Tests various pixel locations and verifies the transformed coordinates
are within the expected working area.
"""

import rclpy
from rclpy.node import Node
from custom_interfaces.srv import PixelToReal
import time


class PixelToRealTester(Node):
    def __init__(self):
        super().__init__('pixel_to_real_tester')
        self.client = self.create_client(PixelToReal, 'pixel_to_real_world')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for pixel_to_real_world service...')
    
    def test_pixel(self, u, v, description=""):
        """Test a single pixel coordinate."""
        request = PixelToReal.Request()
        request.u = u
        request.v = v
        
        self.get_logger().info(f'\n{"="*60}')
        self.get_logger().info(f'Testing: {description}')
        self.get_logger().info(f'Pixel: ({u}, {v})')
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Result: x={response.x:.4f}m, y={response.y:.4f}m, z={response.z:.4f}m')
            
            # Validate working area
            x_valid = -0.50 <= response.x <= 0.50
            y_valid = -0.70 <= response.y <= 0.0
            z_valid = abs(response.z) <= 0.05
            
            status = "✓ PASS" if (x_valid and y_valid and z_valid) else "✗ FAIL"
            self.get_logger().info(f'Validation: {status}')
            self.get_logger().info(f'  X in [-0.50, 0.50]: {x_valid}')
            self.get_logger().info(f'  Y in [-0.70, 0.00]: {y_valid}')
            self.get_logger().info(f'  Z near 0.00: {z_valid}')
            
            return response
        else:
            self.get_logger().error('Service call failed')
            return None
    
    def run_tests(self):
        """Run a comprehensive set of tests."""
        test_cases = [
            (320, 240, "Center of image"),
            (0, 0, "Top-left corner"),
            (640, 0, "Top-right corner"),
            (0, 480, "Bottom-left corner"),
            (640, 480, "Bottom-right corner"),
            (160, 120, "Quarter top-left"),
            (480, 120, "Quarter top-right"),
            (160, 360, "Quarter bottom-left"),
            (480, 360, "Quarter bottom-right"),
            (320, 120, "Top center"),
            (320, 360, "Bottom center"),
            (160, 240, "Left center"),
            (480, 240, "Right center"),
        ]
        
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('STARTING PIXEL-TO-REAL-WORLD TESTS')
        self.get_logger().info('='*60)
        
        results = []
        for u, v, desc in test_cases:
            result = self.test_pixel(u, v, desc)
            results.append((u, v, desc, result))
            time.sleep(0.5)  # Small delay between tests
        
        # Summary
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('TEST SUMMARY')
        self.get_logger().info('='*60)
        
        passed = 0
        failed = 0
        
        for u, v, desc, result in results:
            if result:
                x_valid = -0.50 <= result.x <= 0.50
                y_valid = -0.70 <= result.y <= 0.0
                z_valid = abs(result.z) <= 0.05
                
                if x_valid and y_valid and z_valid:
                    status = "✓"
                    passed += 1
                else:
                    status = "✗"
                    failed += 1
                    
                self.get_logger().info(
                    f'{status} ({u:3d},{v:3d}) -> ({result.x:+.3f}, {result.y:+.3f}, {result.z:+.3f}) | {desc}'
                )
            else:
                failed += 1
                self.get_logger().info(f'✗ ({u:3d},{v:3d}) -> FAILED | {desc}')
        
        self.get_logger().info('='*60)
        self.get_logger().info(f'Results: {passed} passed, {failed} failed out of {len(test_cases)} tests')
        self.get_logger().info('='*60)


def main(args=None):
    rclpy.init(args=args)
    
    tester = PixelToRealTester()
    
    try:
        tester.run_tests()
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
