from launch import LaunchDescription
from launch_ros.actions import Node

import os

def generate_launch_description():
    ld = LaunchDescription()

    show_rgb_image_node = Node(
        package='vision',
        executable='show_rgb_image',
        name='show_rgb_image_node',
        output='screen',
        emulate_tty=True
    )
    ld.add_action(show_rgb_image_node)

    show_depth_image_node = Node(
        package='vision',
        executable='show_depth_image',
        name='show_depth_image_node',
        output='screen',
        emulate_tty=True
    )
    ld.add_action(show_depth_image_node)

    return ld
