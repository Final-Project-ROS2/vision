#!/usr/bin/env python3
"""
Complete SAM Vision Pipeline Launch File for Gazebo Integration
Launches Gazebo world with RGB-D camera and SAM vision pipeline node

Author: ROS2 Vision Pipeline Team
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction,
    RegisterEventHandler
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    FindExecutable,
    Command
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate complete launch description for SAM vision pipeline with Gazebo"""
    
    # Get package directories
    pkg_vision = get_package_share_directory('vision')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # World file path
    world_file = PathJoinSubstitution([
        FindPackageShare('vision'),
        'worlds',
        'sam_vision_world.world'
    ])
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=world_file,
        description='Full path to world file to load'
    )
    
    auto_process_arg = DeclareLaunchArgument(
        'auto_process',
        default_value='true',
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
    
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Start Gazebo GUI'
    )
    
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode'
    )
    
    # Get launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    auto_process = LaunchConfiguration('auto_process')
    processing_rate = LaunchConfiguration('processing_rate')
    save_results = LaunchConfiguration('save_results')
    debug_visualization = LaunchConfiguration('debug_visualization')
    gui = LaunchConfiguration('gui')
    headless = LaunchConfiguration('headless')
    
    # Gazebo server (physics simulation)
    gzserver_cmd = ExecuteProcess(
        cmd=[
            FindExecutable(name='gzserver'),
            '-s', 'libgazebo_ros_factory.so',
            '-s', 'libgazebo_ros_init.so',
            world
        ],
        output='screen',
        condition=UnlessCondition(headless)
    )
    
    # Gazebo client (GUI)
    gzclient_cmd = ExecuteProcess(
        cmd=[FindExecutable(name='gzclient')],
        output='screen',
        condition=IfCondition(gui)
    )
    
    # Static TF publisher for camera frame
    camera_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=[
            '0', '0', '1.5',  # translation x, y, z
            '0', '0.5', '0',  # rotation roll, pitch, yaw
            'world',
            'camera_link'
        ],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # Optical frame (standard camera orientation)
    camera_optical_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_optical_tf_publisher',
        arguments=[
            '0', '0', '0',
            '-0.5', '0.5', '-0.5', '0.5',  # Rotate to optical frame
            'camera_link',
            'camera_link_optical'
        ],
        parameters=[{'use_sim_time': use_sim_time}]
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
                'use_sim_time': use_sim_time
            }
        ],
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),
            ('/camera/depth/image_raw', '/camera/depth/image_raw'),
            ('/camera/camera_info', '/camera/camera_info')
        ]
    )
    
    # Delay SAM node start to allow Gazebo to initialize
    delayed_sam_node = TimerAction(
        period=5.0,
        actions=[sam_pipeline_node]
    )
    
    # Image viewer for debug visualization (optional)
    image_viewer = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='debug_image_viewer',
        arguments=['/vision/debug_image'],
        condition=IfCondition(debug_visualization)
    )
    
    # Robot State Publisher (for TF tree)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': '<robot name="camera"><link name="camera_link"/></robot>'}
        ]
    )
    
    # Joint State Publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # Info messages
    print("\n" + "="*70)
    print("SAM Vision Pipeline with Gazebo Simulation Launch")
    print("="*70)
    print("Pipeline: SAM -> CLIP -> GraspNet -> Scene Understanding")
    print("\nConfiguration:")
    print(f"  World file: {world_file}")
    print(f"  Auto process: {auto_process}")
    print(f"  Processing rate: {processing_rate} Hz")
    print(f"  Save results: {save_results}")
    print(f"  Debug visualization: {debug_visualization}")
    print("\nROS2 Topics:")
    print("  Input:  /camera/image_raw")
    print("  Input:  /camera/depth/image_raw")
    print("  Output: /vision/debug_image")
    print("  Output: /vision/grasp_poses")
    print("\nROS2 Services:")
    print("  /vision/process_scene - Trigger scene processing")
    print("  /vision/reset_pipeline - Reset pipeline state")
    print("\nQuick Commands:")
    print("  Process scene: ros2 service call /vision/process_scene std_srvs/srv/Trigger")
    print("  List topics:   ros2 topic list")
    print("  View image:    ros2 run rqt_image_view rqt_image_view /vision/debug_image")
    print("="*70 + "\n")
    
    return LaunchDescription([
        # Launch arguments
        use_sim_time_arg,
        world_arg,
        auto_process_arg,
        processing_rate_arg,
        save_results_arg,
        debug_visualization_arg,
        gui_arg,
        headless_arg,
        
        # Gazebo
        gzserver_cmd,
        gzclient_cmd,
        
        # TF publishers
        camera_tf_publisher,
        camera_optical_tf_publisher,
        
        # Robot description
        robot_state_publisher,
        joint_state_publisher,
        
        # Vision pipeline (delayed start)
        delayed_sam_node,
        
        # Visualization (optional)
        # image_viewer,  # Uncomment if rqt_image_view is installed
    ])