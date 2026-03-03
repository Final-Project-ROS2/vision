#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.srv import DetectObjects, FindObject, PixelToReal, FindObjectReal



"""
ros2 service call /find_object custom_interfaces/srv/FindObjectReal "{label: 'bowl'}"

confidence calculate by SAM detection based
"""

# TCP_OFFSET = 0.157 # Actual TCP_OFFSET value from teach pendant
TCP_OFFSET = 0.1 # Adjusted TCP_OFFSET


class FindObjectServiceNode(Node):
    def __init__(self):
        super().__init__('find_object_service_node')

        # Parameter toggles tcp offset
        self.declare_parameter('tcp_offset', False)
        self.tcp_offset = bool(self.get_parameter('tcp_offset').value)
        
        # Use reentrant callback group to allow nested service calls
        self.callback_group = ReentrantCallbackGroup()
        
        # Create service server for /find_object
        self.find_object_srv = self.create_service(
            FindObjectReal,
            '/find_object',
            self.find_object_callback,
            callback_group=self.callback_group
        )
        
        # Create service clients for calling other services
        self.detect_objects_client = self.create_client(
            DetectObjects,
            '/vision/detect_objects',
            callback_group=self.callback_group
        )
        
        self.find_object_client = self.create_client(
            FindObject,
            '/vision/find_object',
            callback_group=self.callback_group
        )
        
        self.pixel_to_real_client = self.create_client(
            PixelToReal,
            '/pixel_to_real',
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Find Object Service Node initialized')
        
        # Wait for services to be available
        self.wait_for_services()
    
    def wait_for_services(self):
        """Wait for all required services to be available"""
        self.get_logger().info('Waiting for required services...')
        
        if not self.detect_objects_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/vision/detect_objects service not available')
        
        if not self.find_object_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/vision/find_object service not available')
        
        if not self.pixel_to_real_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/pixel_to_real service not available')
        
        self.get_logger().info('Service clients ready')
    
    def find_object_callback(self, request, response):
        """
        Main service callback for /find_object
        Orchestrates calls to detect_objects, find_object, and pixel_to_real
        """
        label = request.label
        self.get_logger().info(f'Received find_object request for label: {label}')
        
        try:
            # Step 1: Call /vision/detect_objects (synchronously)
            self.get_logger().info('Calling /vision/detect_objects...')
            detect_req = DetectObjects.Request()
            detect_response = self.detect_objects_client.call(detect_req)
            
            if detect_response is None:
                response.success = False
                response.message = 'detect_objects service returned None (service might be unavailable)'
                response.object_id = ''
                response.bbox = []
                response.confidence = 0.0
                response.x = 0.0
                response.y = 0.0
                response.z = 0.0
                self.get_logger().error('detect_objects returned None')
                return response
            
            if not detect_response.success:
                response.success = False
                response.message = f'detect_objects failed: {detect_response.error_message}'
                response.object_id = ''
                response.bbox = []
                response.confidence = 0.0
                response.x = 0.0
                response.y = 0.0
                response.z = 0.0
                return response
            
            self.get_logger().info(f'Detected {detect_response.total_detections} objects')
            
            # Step 2: Call /vision/find_object with the label (synchronously)
            self.get_logger().info(f'Calling /vision/find_object with label: {label}...')
            find_req = FindObject.Request()
            find_req.label = label
            find_response = self.find_object_client.call(find_req)
            
            if find_response is None:
                response.success = False
                response.message = 'find_object service returned None (service might be unavailable)'
                response.object_id = ''
                response.bbox = []
                response.confidence = 0.0
                response.x = 0.0
                response.y = 0.0
                response.z = 0.0
                self.get_logger().error('find_object returned None')
                return response
            
            if not find_response.success:
                response.success = False
                response.message = f'find_object failed: {find_response.message}'
                response.object_id = ''
                response.bbox = []
                response.confidence = 0.0
                response.x = 0.0
                response.y = 0.0
                response.z = 0.0
                return response
            
            # Extract object_id from find_response
            object_id = find_response.object_id if hasattr(find_response, 'object_id') else ''
            
            # Step 3: Calculate center of bounding box (u, v)
            if len(find_response.bbox) < 4:
                response.success = False
                response.message = 'Invalid bounding box returned'
                response.object_id = object_id
                response.bbox = find_response.bbox
                response.confidence = find_response.confidence
                response.x = 0.0
                response.y = 0.0
                response.z = 0.0
                return response
            
            x1, y1, x2, y2 = find_response.bbox[:4]
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)
            
            self.get_logger().info(f'Bounding box center: ({u}, {v})')
            
            # Step 4: Call /pixel_to_real service (synchronously)
            self.get_logger().info('Calling /pixel_to_real...')
            pixel_req = PixelToReal.Request()
            pixel_req.u = u
            pixel_req.v = v
            pixel_response = self.pixel_to_real_client.call(pixel_req)
            
            if pixel_response is None:
                response.success = False
                response.message = 'pixel_to_real service returned None (service might be unavailable)'
                response.object_id = object_id
                response.bbox = find_response.bbox
                response.confidence = find_response.confidence
                response.x = 0.0
                response.y = 0.0
                response.z = 0.0
                self.get_logger().error('pixel_to_real returned None')
                return response
            
            # Step 5: Return final response
            response.success = True
            response.message = f'Successfully found object "{label}" (ID: {object_id}) at ({pixel_response.x:.3f}, {pixel_response.y:.3f}, {pixel_response.z:.3f})'
            response.object_id = object_id
            response.bbox = find_response.bbox
            response.confidence = find_response.confidence
            response.x = pixel_response.x
            response.y = pixel_response.y
            if self.tcp_offset:
                response.z = pixel_response.z + TCP_OFFSET
            else:
                response.z = pixel_response.z
            
            self.get_logger().info(f'Success! Object {object_id} at world coordinates: ({pixel_response.x:.3f}, {pixel_response.y:.3f}, {pixel_response.z:.3f})')
            
        except Exception as e:
            self.get_logger().error(f'Error in find_object_callback: {str(e)}')
            response.success = False
            response.message = f'Internal error: {str(e)}'
            response.object_id = ''
            response.bbox = []
            response.confidence = 0.0
            response.x = 0.0
            response.y = 0.0
            response.z = 0.0
        
        return response


def main(args=None):
    rclpy.init(args=args)
    
    node = FindObjectServiceNode()
    
    # Use MultiThreadedExecutor to handle nested service calls
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        node.get_logger().info('Find Object Service Node spinning...')
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()