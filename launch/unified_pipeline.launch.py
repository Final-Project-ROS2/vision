#!/usr/bin/env python3
"""
Launch file for Unified Vision Pipeline

Starts all required vision nodes:
- simple_sam_detector (SAM detection)
- clip_classifier (CLIP classification)
- graspnet_detector (Grasp detection)
- scene_understanding (Spatial relations)
- pixel_to_real_service (Coordinate conversion)
- unified_pipeline (Pipeline orchestrator)

Usage:
    ros2 launch vision unified_pipeline.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level (debug, info, warn, error)'
        ),
        
        # 1. SAM Detector (core detection)
        Node(
            package='vision',
            executable='simple_sam_detector',
            name='simple_sam_detector',
            output='screen',
            parameters=[{
                'use_sim_time': False,
            }],
            arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
        ),
        
        # 2. CLIP Classifier (classification) - delay 2s for SAM to initialize
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='vision',
                    executable='clip_classifier',
                    name='clip_classifier',
                    output='screen',
                    parameters=[{
                        'use_sim_time': False,
                    }],
                    arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
                ),
            ]
        ),
        
        # 3. GraspNet Detector (grasp detection) - delay 4s
        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package='vision',
                    executable='graspnet_detector',
                    name='graspnet_detector',
                    output='screen',
                    parameters=[{
                        'use_sim_time': False,
                    }],
                    arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
                ),
            ]
        ),
        
        # 4. Scene Understanding (spatial relations) - delay 6s
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='vision',
                    executable='scene_understanding',
                    name='scene_understanding',
                    output='screen',
                    parameters=[{
                        'use_sim_time': False,
                    }],
                    arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
                ),
            ]
        ),
        
        # 5. Pixel-to-Real Service (coordinate conversion) - delay 8s
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package='vision',
                    executable='pixel_to_real_service',
                    name='pixel_to_real_service',
                    output='screen',
                    parameters=[{
                        'use_sim_time': False,
                    }],
                    arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
                ),
            ]
        ),
        
        # 6. Unified Pipeline (orchestrator) - delay 10s to ensure all services are ready
        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='vision',
                    executable='unified_pipeline',
                    name='unified_pipeline',
                    output='screen',
                    parameters=[{
                        'use_sim_time': False,
                    }],
                    arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
                ),
            ]
        ),
    ])
