"""
ROS 2 Launch File: Insta360 camera + undistorted UNet / Pix2Pix crack detection for one lens

Usage:
    ros2 launch Agent_md/undistortion_pipeline.launch.py                 # front lens
    ros2 launch Agent_md/undistortion_pipeline.launch.py lens:=back
    ros2 launch Agent_md/undistortion_pipeline.launch.py use_pix2pix:=true undistort_balance:=0.3

Nodes Started:
    1. insta360_publisher — publishes /insta360/{front,back}/image_raw from /dev/video0
    2. crack_detection_node (via crack_detection.launch.py) — subscribes to the chosen lens,
       undistorts with calibration_data/insta360_oner_<lens>.yaml, runs UNet (+ Pix2Pix)

Undistortion happens inside crack_detection_node (see claude.md TASK 3+4); there is no
separate undistortion node. To keep using this, copy it to crack_detection/launch/.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_detection(context):
    lens = LaunchConfiguration('lens').perform(context)
    if lens not in ('front', 'back'):
        raise RuntimeError(f"lens must be 'front' or 'back', got {lens!r}")

    detection_launch = os.path.join(
        get_package_share_directory('crack_detection'), 'launch', 'crack_detection.launch.py')
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(detection_launch),
        launch_arguments={
            'camera_topic': f'/insta360/{lens}/image_raw',
            'calibration_file': f'calibration_data/insta360_oner_{lens}.yaml',
            'undistort_enabled': 'true',
            'undistort_balance': LaunchConfiguration('undistort_balance').perform(context),
            'use_pix2pix': LaunchConfiguration('use_pix2pix').perform(context),
            'use_tiling': LaunchConfiguration('use_tiling').perform(context),
        }.items(),
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('lens', default_value='front',
                              description="Lens to run detection on: 'front' or 'back'"),
        DeclareLaunchArgument('device', default_value='/dev/video0',
                              description='Insta360 ONE R V4L2 device (USB webcam mode)'),
        DeclareLaunchArgument('undistort_balance', default_value='0.5',
                              description='0 = crop to valid pixels, 1 = keep full field of view'),
        DeclareLaunchArgument('use_pix2pix', default_value='false',
                              description='Refine UNet masks with Pix2Pix'),
        DeclareLaunchArgument('use_tiling', default_value='false',
                              description='Tiled inference (higher accuracy, slower)'),

        Node(
            package='insta360_ros',
            executable='insta360_publisher',
            name='insta360_publisher',
            output='screen',
            parameters=[{'device': LaunchConfiguration('device')}],
        ),
        OpaqueFunction(function=launch_detection),
    ])
