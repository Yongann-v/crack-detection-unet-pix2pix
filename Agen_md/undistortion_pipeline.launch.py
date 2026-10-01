"""
ROS 2 Launch File: Insta360 Undistortion Pipeline

Usage:
    ros2 launch crack_detection undistortion_pipeline.launch.py

Nodes Started:
    1. /insta360_undistortion_node — Subscribes /camera/image_raw, publishes /camera/undistorted
    2. /yolov8_detection_node (optional) — Consumes /camera/undistorted, publishes detections
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    
    # Declare launch arguments
    calib_dir_arg = DeclareLaunchArgument(
        'calib_dir',
        default_value=os.path.join(os.path.expanduser('~'), 'robot_ws', 'src', 'crack_detection', 'calibration_data'),
        description='Path to calibration data directory (contains insta360_oner_K.npy, insta360_oner_D.npy)'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Input camera topic (raw Insta360 images)'
    )
    
    undistorted_topic_arg = DeclareLaunchArgument(
        'undistorted_topic',
        default_value='/camera/undistorted',
        description='Output undistorted image topic'
    )
    
    jetson_mode_arg = DeclareLaunchArgument(
        'jetson_mode',
        default_value='true',
        description='Enable Jetson Xavier NX optimizations (GPU, TensorRT)'
    )
    
    # Get launch configuration
    calib_dir = LaunchConfiguration('calib_dir')
    camera_topic = LaunchConfiguration('camera_topic')
    undistorted_topic = LaunchConfiguration('undistorted_topic')
    jetson_mode = LaunchConfiguration('jetson_mode')
    
    # Undistortion Node
    undistortion_node = Node(
        package='crack_detection',
        executable='undistort_node.py',
        name='insta360_undistortion_node',
        output='screen',
        parameters=[
            {
                'calib_dir': calib_dir,
                'camera_topic': camera_topic,
                'undistorted_topic': undistorted_topic,
                'jetson_mode': jetson_mode,
                'pole_crop_percent': 0.1,  # Skip top/bottom 10%
                'frame_queue_size': 5,  # Small buffer to reduce latency
                'image_encoding': 'bgr8',
            }
        ],
        remappings=[
            ('image_raw', camera_topic),
            ('image_undistorted', undistorted_topic),
        ]
    )
    
    # Optional: YOLOv8 Detection Node (consumers /camera/undistorted)
    # Uncomment if you want to run detection in same launch file
    # yolov8_node = Node(
    #     package='crack_detection',
    #     executable='yolov8_detection_node.py',
    #     name='yolov8_defect_detection',
    #     output='screen',
    #     parameters=[
    #         {
    #             'model_path': 'models/yolov8n-defects.pt',
    #             'confidence_threshold': 0.6,
    #             'jetson_mode': jetson_mode,
    #         }
    #     ],
    #     remappings=[
    #         ('image_in', undistorted_topic),
    #         ('detections_out', '/defects/detections'),
    #     ]
    # )
    
    return LaunchDescription([
        # Launch arguments
        calib_dir_arg,
        camera_topic_arg,
        undistorted_topic_arg,
        jetson_mode_arg,
        
        # Nodes
        undistortion_node,
        # yolov8_node,  # Uncomment to enable
    ])
