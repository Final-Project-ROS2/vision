from launch import LaunchDescription
from launch_ros.actions import Node

import os

def generate_launch_description():
    ld = LaunchDescription()

    simple_sam_detector_node = Node(
        package='vision',
        executable='simple_sam_detector',
        name='simple_sam_detector_node',
        output='screen',
        emulate_tty=True
    )
    ld.add_action(simple_sam_detector_node)

    clip_classifier_node = Node(
        package='vision',
        executable='clip_classifier',
        name='clip_classifier_node',
        output='screen',
        emulate_tty=True
    )
    ld.add_action(clip_classifier_node)

    graspnet_detector_node = Node(
        package='vision',
        executable='graspnet_detector',
        name='graspnet_detector_node',
        output='screen',
        emulate_tty=True
    )
    ld.add_action(graspnet_detector_node)

    scene_understanding_node = Node(
        package='vision',
        executable='scene_understanding',
        name='scene_understanding_node',
        output='screen',
        emulate_tty=True
    )
    ld.add_action(scene_understanding_node)

    return ld
