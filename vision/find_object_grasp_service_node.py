#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.srv import FindObjectGrasp, FindObjectReal


class FindObjectGraspServiceNode(Node):
    def __init__(self):
        super().__init__('find_object_grasp_service_node')
        
        # Use reentrant callback group to allow nested service calls
        self.callback_group = ReentrantCallbackGroup()
        
        # Create service server for /find_object
        self.find_object_grasp_srv = self.create_service(
            FindObjectGrasp,
            '/find_object_grasp',
            self.find_object_grasp_callback,
            callback_group=self.callback_group
        )
        
        # Create service clients for calling other services
        self.find_object_client = self.create_client(
            FindObjectReal,
            '/find_object',
            callback_group=self.callback_group
        )
        
        self.get_logger().info('Find Object Grasp Service Node initialized')
        
        # Wait for services to be available
        self.wait_for_services()
    
    def wait_for_services(self):
        """Wait for all required services to be available"""
        self.get_logger().info('Waiting for required services...')
        
        if not self.find_object_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/find_object service not available')
        
        self.get_logger().info('Service clients ready')
    
    def find_object_grasp_callback(self, request, response):
        """
        Main service callback for /find_object_grasp
        Returns the bounding box of the requested object
        """
        label = request.label
        self.get_logger().info(f'Received find_object_grasp request for label: {label}')
        
        try:
            # Call /find_object with the label (synchronously)
            self.get_logger().info(f'Calling /find_object with label: {label}...')
            find_req = FindObjectReal.Request()
            find_req.label = label
            find_response = self.find_object_client.call(find_req)
            
            if not find_response.success:
                response.success = False
                response.error_message = f'find_object failed: {find_response.message}'
                return response
            
            # Return the bounding box in the grasp_pose structure
            response.success = True
            response.grasp_pose.object_id = find_response.object_id
            response.grasp_pose.bbox = find_response.bbox
            response.error_message = ''
            
            self.get_logger().info(f'Successfully found {label}: bbox={find_response.bbox}')
            
        except Exception as e:
            self.get_logger().error(f'Exception in find_object_grasp_callback: {e}')
            response.success = False
            response.error_message = str(e)
        
        return response


def main(args=None):
    rclpy.init(args=args)
    
    node = FindObjectGraspServiceNode()
    
    # Use MultiThreadedExecutor to handle nested service calls
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        node.get_logger().info('Find Object Grasp Service Node spinning...')
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()