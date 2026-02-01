#!/usr/bin/env python3
"""
Test Client for Scene Understanding Service

This script acts as a client to call the /vision/understand_scene service
and prints the returned scene analysis.

Services Called:
    1. /vision/understand_scene
       - Service Type: std_srvs/srv/Trigger
       - Action: Requests the scene_understanding_node to analyze the current scene.
       - Response: A JSON string containing detailed information about detected
                   objects, their spatial relationships, and graspability.

How to Run:
    1. Ensure the full vision pipeline is running. You can use the provided
       launch file or run the nodes manually.
       - Terminal 1 (Gazebo Sim with camera):
         ros2 launch ur_yt_sim spawn_ur5_camera_gripper_moveit.launch.py
       - Terminal 2 (SAM Detector):
         ros2 run vision simple_sam_detector
       - Terminal 3 (CLIP Classifier):
         ros2 run vision clip_classifier
       - Terminal 4 (GraspNet Detector):
         ros2 run vision graspnet_detector
       - Terminal 5 (Scene Understanding Node):
         ros2 run vision scene_understanding

    2. In a new terminal, run this test client:
       ros2 run vision test_scene_service

Expected Output:
    The script will print the JSON response from the /vision/understand_scene
    service, which includes details about each object, its relations to other
    objects, and whether it is graspable. If the service call fails, it will
    print an error message.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import json

class SceneServiceTestClient(Node):
    """
    A simple client node to test the /vision/understand_scene service.
    """
    def __init__(self):
        super().__init__('scene_service_test_client')
        self.client = self.create_client(Trigger, '/vision/understand_scene')
        
        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /vision/understand_scene not available, waiting again...')
        
        self.request = Trigger.Request()

    def call_service(self):
        """
        Calls the /vision/understand_scene service and waits for the response.
        """
        self.get_logger().info("Calling /vision/understand_scene service...")
        future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, future)
        
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Service call successful!")
                # The response message is a JSON string, parse it for pretty printing
                try:
                    scene_data = json.loads(response.message)
                    self.get_logger().info("Scene Analysis Result:\n" + json.dumps(scene_data, indent=2))
                except json.JSONDecodeError:
                    self.get_logger().error("Failed to parse JSON response: " + response.message)
            else:
                self.get_logger().error(f"Service call failed with message: {response.message}")
        except Exception as e:
            self.get_logger().error(f'Service call failed with exception: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    test_client = SceneServiceTestClient()
    test_client.call_service()
    
    test_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
