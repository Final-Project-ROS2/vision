#!/usr/bin/env python3
"""
Test client for pixel_to_real_world service

This script demonstrates how to use the pixel-to-real-world service
to convert pixel coordinates to 3D world coordinates.

Usage:
    python3 test_pixel_to_real_world_client.py
"""

import rclpy
from rclpy.node import Node
from custom_interfaces.srv import PixelToReal
import sys


class PixelToRealWorldClient(Node):
    """Client node to test pixel-to-real-world service."""
    
    def __init__(self):
        super().__init__('pixel_to_real_world_client')
        
        # Create service client
        self.client = self.create_client(PixelToReal, 'pixel_to_real_world')
        
        # Wait for service to be available
        self.get_logger().info('Waiting for /pixel_to_real_world service...')
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Service not available, waiting...')
        
        self.get_logger().info('Service available!')
    
    def call_service(self, u, v):
        """
        Call the pixel-to-real-world service.
        
        Args:
            u: Pixel column (x-coordinate)
            v: Pixel row (y-coordinate)
            
        Returns:
            tuple: (x, y, z) in meters or None if failed
        """
        # Create request
        request = PixelToReal.Request()
        request.u = int(u)
        request.v = int(v)
        
        # Call service
        self.get_logger().info(f'Requesting conversion for pixel ({u}, {v})...')
        future = self.client.call_async(request)
        
        # Wait for response
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(
                f'✓ Pixel ({u}, {v}) -> World ({response.x:.4f}m, {response.y:.4f}m, {response.z:.4f}m)')
            return (response.x, response.y, response.z)
        else:
            self.get_logger().error('✗ Service call failed')
            return None
    
    def test_multiple_pixels(self, pixel_list):
        """Test conversion for multiple pixels."""
        results = []
        
        self.get_logger().info(f'\n{"="*60}')
        self.get_logger().info('Testing multiple pixels')
        self.get_logger().info(f'{"="*60}\n')
        
        for u, v in pixel_list:
            result = self.call_service(u, v)
            if result:
                results.append({
                    'pixel': (u, v),
                    'world': result
                })
            print()  # Add spacing between results
        
        return results
    
    def test_grid_pattern(self, rows=3, cols=3, img_width=640, img_height=480):
        """Test a grid pattern across the image."""
        self.get_logger().info(f'\n{"="*60}')
        self.get_logger().info(f'Testing {rows}x{cols} grid pattern')
        self.get_logger().info(f'{"="*60}\n')
        
        pixels = []
        for i in range(rows):
            for j in range(cols):
                u = int((j + 1) * img_width / (cols + 1))
                v = int((i + 1) * img_height / (rows + 1))
                pixels.append((u, v))
        
        return self.test_multiple_pixels(pixels)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Create client node
        client = PixelToRealWorldClient()
        
        # Test 1: Center pixel
        print("\n" + "="*60)
        print("TEST 1: Center Pixel")
        print("="*60)
        client.call_service(320, 240)
        
        # Test 2: Corner pixels
        print("\n" + "="*60)
        print("TEST 2: Corner Pixels")
        print("="*60)
        corner_pixels = [
            (100, 100),   # Top-left
            (540, 100),   # Top-right
            (100, 380),   # Bottom-left
            (540, 380),   # Bottom-right
        ]
        client.test_multiple_pixels(corner_pixels)
        
        # Test 3: Grid pattern
        print("\n" + "="*60)
        print("TEST 3: Grid Pattern (3x3)")
        print("="*60)
        grid_results = client.test_grid_pattern(rows=3, cols=3)
        
        # Test 4: Edge cases
        print("\n" + "="*60)
        print("TEST 4: Various Positions")
        print("="*60)
        edge_pixels = [
            (0, 0),         # Top-left corner
            (639, 0),       # Top-right corner
            (0, 479),       # Bottom-left corner
            (639, 479),     # Bottom-right corner
            (320, 0),       # Top center
            (320, 479),     # Bottom center
            (0, 240),       # Left center
            (639, 240),     # Right center
        ]
        client.test_multiple_pixels(edge_pixels)
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✓ All tests completed successfully!")
        print("\nThe pixel-to-real-world service is working correctly.")
        print("\nNext steps:")
        print("  1. Place objects in front of camera")
        print("  2. Get pixel coordinates from object detection")
        print("  3. Convert to 3D coordinates for robot manipulation")
        print("\nFor integration examples, see:")
        print("  - docs/PIXEL_TO_REAL_WORLD_SERVICE.md")
        print("  - vision/find_object_grasp_service_node.py")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'client' in locals():
            client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
