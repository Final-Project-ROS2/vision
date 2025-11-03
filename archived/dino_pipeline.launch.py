#!/usr/bin/env python3
"""
ROS2 Launch file for SAM Vision Pipeline with Gazebo simulation
Launches the complete vision pipeline with camera simulation and visualization

Pipeline: SAM (Meta) → CLIP → GraspNet → Scene Understanding
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from pathlib import Path


def generate_launch_description():
    """Generate launch description for SAM vision pipeline"""
    
    # Declare launch arguments
    use_gazebo_arg = DeclareLaunchArgument(
        'use_gazebo',
        default_value='true',
        description='Whether to launch Gazebo simulation'
    )
    
    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value='empty.world',
        description='Gazebo world file to load'
    )
    
    auto_process_arg = DeclareLaunchArgument(
        'auto_process',
        default_value='false',
        description='Enable automatic scene processing'
    )
    
    processing_rate_arg = DeclareLaunchArgument(
        'processing_rate',
        default_value='1.0',
        description='Scene processing rate in Hz'
    )
    
    save_results_arg = DeclareLaunchArgument(
        'save_results',
        default_value='true',
        description='Save pipeline results to disk'
    )
    
    debug_visualization_arg = DeclareLaunchArgument(
        'debug_visualization',
        default_value='true',
        description='Enable debug visualization'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera',
        description='Base camera topic namespace'
    )
    
    # Get launch configurations
    use_gazebo = LaunchConfiguration('use_gazebo')
    world_file = LaunchConfiguration('world_file')
    auto_process = LaunchConfiguration('auto_process')
    processing_rate = LaunchConfiguration('processing_rate')
    save_results = LaunchConfiguration('save_results')
    debug_visualization = LaunchConfiguration('debug_visualization')
    camera_topic = LaunchConfiguration('camera_topic')
    
    # Gazebo launch (if enabled)
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world_file,
            'pause': 'false',
            'gui': 'true',
            'use_sim_time': 'true'
        }.items(),
        condition=IfCondition(use_gazebo)
    )
    
    # Robot state publisher (for TF frames)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ],
        condition=IfCondition(use_gazebo)
    )
    
    # Camera spawner (for Gazebo simulation)
    # This will spawn a simple RGB-D camera in Gazebo
    camera_spawner = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
            '-entity', 'camera_sensor',
            '-topic', '/robot_description',
            '-x', '0', '-y', '0', '-z', '1',
            '-R', '0', '-P', '0', '-Y', '0'
        ],
        output='screen',
        condition=IfCondition(use_gazebo)
    )
    
    # SAM Vision Pipeline Node
    sam_pipeline_node = Node(
        package='vision',
        executable='sam_vision_pipeline',
        name='sam_vision_pipeline',
        output='screen',
        parameters=[
            {
                'auto_process': auto_process,
                'processing_rate': processing_rate,
                'save_results': save_results,
                'debug_visualization': debug_visualization,
                'use_sim_time': use_gazebo
            }
        ],
        remappings=[
            ('/camera/image_raw', [camera_topic, '/image_raw']),
            ('/camera/depth/image_raw', [camera_topic, '/depth/image_raw']),
            ('/camera/camera_info', [camera_topic, '/camera_info'])
        ]
    )
    
    # Image view for debugging (optional)
    image_viewer = Node(
        package='image_view',
        executable='image_view',
        name='debug_image_viewer',
        output='screen',
        remappings=[
            ('image', '/vision/debug_image')
        ],
        condition=IfCondition(debug_visualization)
    )
    
    # RViz for visualization
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('vision'),
        'config',
        'dino_pipeline.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[
            {'use_sim_time': use_gazebo}
        ]
    )
    
    # Static transform publisher for camera frame
    camera_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=[
            '0', '0', '1',  # translation x, y, z
            '0', '0', '0', '1',  # rotation x, y, z, w
            'world',  # parent frame
            'camera_link'  # child frame
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        use_gazebo_arg,
        world_file_arg,
        auto_process_arg,
        processing_rate_arg,
        save_results_arg,
        debug_visualization_arg,
        camera_topic_arg,
        
        # Gazebo simulation (conditional)
        gazebo_launch,
        robot_state_publisher,
        camera_spawner,
        
        # Core pipeline components
        sam_pipeline_node,
        camera_tf_publisher,
        
        # Visualization (conditional)
        image_viewer,
        rviz_node,
    ])