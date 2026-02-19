#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.srv import FindObjectGrasp, FindObjectReal, DetectGraspBBox

TCP_OFFSET = 0.1

class FindObjectGraspServiceNode(Node):
    def __init__(self):
        super().__init__('find_object_grasp_service_node')

        # Parameter toggles tcp offset
        self.declare_parameter('tcp_offset', False)
        self.tcp_offset = bool(self.get_parameter('tcp_offset').value)
        
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
        
        self.detect_grasp_bbox_client = self.create_client(
            DetectGraspBBox,
            '/vision/detect_grasp_bb',
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
        
        if not self.detect_grasp_bbox_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/vision/detect_grasp_bb service not available')
        
        self.get_logger().info('Service clients ready')
    
    def find_object_grasp_callback(self, request, response):
        """
        Main service callback for /find_object_grasp
        Orchestrates calls to find_object and detect_grasp
        """
        label = request.label
        self.get_logger().info(f'Received find_object_grasp request for label: {label}')
        
        try:
            # Step 1: Call /find_object with the label (synchronously)
            self.get_logger().info(f'Calling /find_object with label: {label}...')
            find_req = FindObjectReal.Request()
            find_req.label = label
            find_response = self.find_object_client.call(find_req)
            
            if not find_response.success:
                response.success = False
                response.error_message = f'find_object failed: {find_response.message}'
                return response
            
            # Step 2: Call /vision/detect_grasp_bb service (synchronously)
            self.get_logger().info('Calling /vision/detect_grasp_bb...')
            detect_grasp_bbox_req = DetectGraspBBox.Request()
            detect_grasp_bbox_req.x1 = find_response.bbox[0]
            detect_grasp_bbox_req.y1 = find_response.bbox[1]
            detect_grasp_bbox_req.x2 = find_response.bbox[2]
            detect_grasp_bbox_req.y2 = find_response.bbox[3]
            self.get_logger().info(f'BBox for grasp detection: {detect_grasp_bbox_req.x1}, {detect_grasp_bbox_req.y1}, {detect_grasp_bbox_req.x2}, {detect_grasp_bbox_req.y2}')
            detect_grasp_bbox_response = self.detect_grasp_bbox_client.call(detect_grasp_bbox_req)

            if not detect_grasp_bbox_response.success:
                response.success = False
                response.error_message = f'find_object failed: {find_response.message}'
                return response

            response.success = True
            if self.tcp_offset:
                detect_grasp_bbox_response.grasp_pose.position.z += TCP_OFFSET
            response.grasp_pose = detect_grasp_bbox_response.grasp_pose
            response.error_message = ''
            
            
            
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