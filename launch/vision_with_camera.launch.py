#!/usr/bin/env python3
"""
Launch file for complete vision system with camera service

This launches:
1. Camera Service Node (opens webcam/depth camera)
2. SAM Vision Pipeline Node (processes images)

Usage:
    # Webcam
    ros2 launch vision vision_with_camera.launch.py
    
    # RealSense
    ros2 launch vision vision_with_camera.launch.py camera_type:=realsense
    
    # Test image
    ros2 launch vision vision_with_camera.launch.py camera_type:=file image_file:=Final-proj/src/arrange.jpg
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for complete vision system"""
    
    # Declare launch arguments
    camera_type_arg = DeclareLaunchArgument(
        'camera_type',
        default_value='webcam',
        description='Camera type: webcam, realsense, or file'
    )
    
    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Camera device ID (for webcam)'
    )
    
    image_file_arg = DeclareLaunchArgument(
        'image_file',
        default_value='',
        description='Path to image/video file (for file type)'
    )
    
    width_arg = DeclareLaunchArgument(
        'width',
        default_value='640',
        description='Image width in pixels'
    )
    
    height_arg = DeclareLaunchArgument(
        'height',
        default_value='480',
        description='Image height in pixels'
    )
    
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30.0',
        description='Camera frame rate'
    )
    
    auto_process_arg = DeclareLaunchArgument(
        'auto_process',
        default_value='false',
        description='Auto-process incoming frames in vision pipeline'
    )
    
    # Camera Service Node
    camera_node = Node(
        package='vision',
        executable='camera_service',
        name='camera_service',
        output='screen',
        parameters=[{
            'camera_type': LaunchConfiguration('camera_type'),
            'camera_id': LaunchConfiguration('camera_id'),
            'image_file': LaunchConfiguration('image_file'),
            'width': LaunchConfiguration('width'),
            'height': LaunchConfiguration('height'),
            'fps': LaunchConfiguration('fps'),
            'auto_start': True
        }]
    )
    
    # SAM Vision Pipeline Node
    vision_pipeline_node = Node(
        package='vision',
        executable='sam_vision_pipeline',
        name='sam_vision_pipeline',
        output='screen',
        parameters=[{
            'auto_process': LaunchConfiguration('auto_process'),
            'save_results': True,
            'debug_visualization': True,
            'processing_rate': 1.0
        }]
    )
    
    return LaunchDescription([
        # Launch arguments
        camera_type_arg,
        camera_id_arg,
        image_file_arg,
        width_arg,
        height_arg,
        fps_arg,
        auto_process_arg,
        
        # Nodes
        camera_node,
        vision_pipeline_node,
    ])
