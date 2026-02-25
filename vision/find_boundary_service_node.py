"""
Service: FindBoundary

Request:
- label: string

Response:
- success: bool
- message: string
- object_id: string
- bbox: list[float] (original bounding box)
- confidence: float
- x1: float (real-world x of bbox[0], y1: real-world y of bbox[1])
- y1: float
- x2: float (real-world x of bbox[2], y2: real-world y of bbox[3])
- y2: float

Description:
Given an object label, this service finds the object using /vision/find_object, then converts the first two bbox coordinates (x1, y1) and the last two (x2, y2) to real-world coordinates using /pixel_to_real. Returns both points in real-world coordinates.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from custom_interfaces.srv import DetectObjects, FindObject, PixelToReal

class FindBoundaryServiceNode(Node):
    def __init__(self):
        super().__init__('find_boundary_service_node')

        self.callback_group = ReentrantCallbackGroup()

        self.find_boundary_srv = self.create_service(
            FindBoundary,  # Custom interface to be defined
            '/find_boundary',
            self.find_boundary_callback,
            callback_group=self.callback_group
        )

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
        self.get_logger().info('Find Boundary Service Node initialized')
        self.wait_for_services()

    def wait_for_services(self):
        self.get_logger().info('Waiting for required services...')
        if not self.detect_objects_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/vision/detect_objects service not available')
        if not self.find_object_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/vision/find_object service not available')
        if not self.pixel_to_real_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/pixel_to_real service not available')
        self.get_logger().info('Service clients ready')

    def find_boundary_callback(self, request, response):
        label = request.label
        self.get_logger().info(f'Received find_boundary request for label: {label}')
        try:
            self.get_logger().info('Calling /vision/detect_objects...')
            detect_req = DetectObjects.Request()
            detect_response = self.detect_objects_client.call(detect_req)
            if detect_response is None or not detect_response.success:
                response.success = False
                response.message = 'detect_objects failed'
                response.object_id = ''
                response.bbox = []
                response.confidence = 0.0
                response.x1 = 0.0
                response.y1 = 0.0
                response.x2 = 0.0
                response.y2 = 0.0
                return response
            self.get_logger().info('Calling /vision/find_object...')
            find_req = FindObject.Request()
            find_req.label = label
            find_response = self.find_object_client.call(find_req)
            if find_response is None or not find_response.success:
                response.success = False
                response.message = 'find_object failed'
                response.object_id = ''
                response.bbox = []
                response.confidence = 0.0
                response.x1 = 0.0
                response.y1 = 0.0
                response.x2 = 0.0
                response.y2 = 0.0
                return response
            object_id = getattr(find_response, 'object_id', '')
            bbox = find_response.bbox
            if len(bbox) < 4:
                response.success = False
                response.message = 'Invalid bounding box returned'
                response.object_id = object_id
                response.bbox = bbox
                response.confidence = find_response.confidence
                response.x1 = 0.0
                response.y1 = 0.0
                response.x2 = 0.0
                response.y2 = 0.0
                return response
            x1, y1, x2, y2 = bbox[:4]
            self.get_logger().info(f'Converting bbox points: ({x1}, {y1}), ({x2}, {y2})')
            pixel_req1 = PixelToReal.Request()
            pixel_req1.u = int(x1)
            pixel_req1.v = int(y1)
            pixel_resp1 = self.pixel_to_real_client.call(pixel_req1)
            pixel_req2 = PixelToReal.Request()
            pixel_req2.u = int(x2)
            pixel_req2.v = int(y2)
            pixel_resp2 = self.pixel_to_real_client.call(pixel_req2)
            if pixel_resp1 is None or pixel_resp2 is None:
                response.success = False
                response.message = 'pixel_to_real failed'
                response.object_id = object_id
                response.bbox = bbox
                response.confidence = find_response.confidence
                response.x1 = 0.0
                response.y1 = 0.0
                response.x2 = 0.0
                response.y2 = 0.0
                return response
            response.success = True
            response.message = f'Successfully found boundary for "{label}" (ID: {object_id})'
            response.object_id = object_id
            response.bbox = bbox
            response.confidence = find_response.confidence
            response.x1 = pixel_resp1.x
            response.y1 = pixel_resp1.y
            response.x2 = pixel_resp2.x
            response.y2 = pixel_resp2.y
            self.get_logger().info(f'Boundary points: ({response.x1}, {response.y1}), ({response.x2}, {response.y2})')
        except Exception as e:
            self.get_logger().error(f'Error in find_boundary_callback: {str(e)}')
            response.success = False
            response.message = f'Internal error: {str(e)}'
            response.object_id = ''
            response.bbox = []
            response.confidence = 0.0
            response.x1 = 0.0
            response.y1 = 0.0
            response.x2 = 0.0
            response.y2 = 0.0
        return response

def main(args=None):
    rclpy.init(args=args)
    node = FindBoundaryServiceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        node.get_logger().info('Find Boundary Service Node spinning...')
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
