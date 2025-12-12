#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get path to the YAML config file
    config_file = os.path.join(
        get_package_share_directory('crack_detection'),
        'config',
        'crack_detection_params.yaml'
    )
    
    # Declare launch arguments (these will OVERRIDE YAML values when specified)
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/best_model.pth',
        description='Path to trained UNet model'
    )
    
    pix2pix_model_path_arg = DeclareLaunchArgument(
        'pix2pix_model_path',
        default_value='models/pix2pix_epoch_98_best.pth',
        description='Path to trained Pix2Pix model'
    )
    
    use_tiling_arg = DeclareLaunchArgument(
        'use_tiling',
        default_value='false',
        description='Use tiled inference (slower but more accurate)'
    )
    
    use_pix2pix_arg = DeclareLaunchArgument(
        'use_pix2pix',
        default_value='false',
        description='Use Pix2Pix refinement (better quality)'
    )
    
    threshold_arg = DeclareLaunchArgument(
        'threshold',
        default_value='0.5',
        description='Detection threshold (0.0-1.0)'
    )
    
    min_crack_percent_arg = DeclareLaunchArgument(
        'min_crack_percent',
        default_value='2.0',  # Match YAML default
        description='Minimum crack percentage to trigger detection'
    )
    
    skip_frames_arg = DeclareLaunchArgument(
        'skip_frames',
        default_value='1',
        description='Process every Nth frame (1=all frames)'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/a200_1103/sensors/camera_0/color/image',
        description='Camera image topic to subscribe to'
    )
    
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/a200_1103/sensors/camera_0/depth/image',
        description='Depth image topic to subscribe to'
    )
    
    # Zoom arguments
    zoom_enabled_arg = DeclareLaunchArgument(
        'zoom_enabled',
        default_value='false',
        description='Enable zoom feature'
    )
    
    zoom_factor_arg = DeclareLaunchArgument(
        'zoom_factor',
        default_value='2.0',
        description='Zoom magnification factor'
    )
    
    zoom_center_x_arg = DeclareLaunchArgument(
        'zoom_center_x',
        default_value='0.5',
        description='Horizontal zoom center (0.0-1.0)'
    )
    
    zoom_center_y_arg = DeclareLaunchArgument(
        'zoom_center_y',
        default_value='0.5',
        description='Vertical zoom center (0.0-1.0)'
    )
    
    save_directory_arg = DeclareLaunchArgument(
        'save_directory',
        default_value='crack_captures',
        description='Directory to save captured images'
    )
    
    # Temporal consistency arguments
    temporal_window_size_arg = DeclareLaunchArgument(
        'temporal_window_size',
        default_value='5',  # Match YAML default
        description='Number of consecutive frames for temporal consistency check'
    )
    
    temporal_threshold_arg = DeclareLaunchArgument(
        'temporal_threshold',
        default_value='4',  # Match YAML default
        description='Minimum frames that must detect crack within temporal window'
    )
    
    # Hysteresis arguments
    hysteresis_enabled_arg = DeclareLaunchArgument(
        'hysteresis_enabled',
        default_value='true',  # Match YAML default
        description='Enable hysteresis to prevent detection flickering'
    )
    
    hysteresis_low_threshold_arg = DeclareLaunchArgument(
        'hysteresis_low_threshold',
        default_value='1.0',  # Match YAML default
        description='Lower threshold for turning OFF detection (percentage)'
    )
    
    hysteresis_min_duration_arg = DeclareLaunchArgument(
        'hysteresis_min_duration',
        default_value='0.5',
        description='Minimum detection duration in seconds'
    )
    
    # Depth filtering arguments
    depth_filtering_enabled_arg = DeclareLaunchArgument(
        'depth_filtering_enabled',
        default_value='true',  # Match YAML default
        description='Enable depth-based filtering to remove non-wall detections'
    )
    
    use_adaptive_depth_arg = DeclareLaunchArgument(
        'use_adaptive_depth',
        default_value='true',  # Match YAML default
        description='Automatically adapt to current wall distance (recommended for moving inspection)'
    )
    
    target_inspection_distance_arg = DeclareLaunchArgument(
        'target_inspection_distance',
        default_value='1.5',
        description='Target wall distance in meters (used if adaptive depth disabled)'
    )
    
    depth_tolerance_arg = DeclareLaunchArgument(
        'depth_tolerance',
        default_value='0.1',
        description='Depth tolerance in meters (±tolerance around target distance)'
    )
    
    # Create the crack detection node
    crack_detection_node = Node(
        package='crack_detection',
        executable='crack_detection_node',
        name='crack_detection_node',
        output='screen',
        parameters=[
            config_file,  # Load YAML file FIRST (base configuration)
            {
                # These will OVERRIDE YAML values
                'model_path': LaunchConfiguration('model_path'),
                'pix2pix_model_path': LaunchConfiguration('pix2pix_model_path'),
                'use_tiling': LaunchConfiguration('use_tiling'),
                'use_pix2pix': LaunchConfiguration('use_pix2pix'),
                'threshold': LaunchConfiguration('threshold'),
                'min_crack_percent': LaunchConfiguration('min_crack_percent'),
                'skip_frames': LaunchConfiguration('skip_frames'),
                'camera_topic': LaunchConfiguration('camera_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'input_size': 384,
                'window_size': 384,
                'subdivisions': 2,
                'publish_visualization': True,
                # Zoom parameters
                'zoom_enabled': LaunchConfiguration('zoom_enabled'),
                'zoom_factor': LaunchConfiguration('zoom_factor'),
                'zoom_center_x': LaunchConfiguration('zoom_center_x'),
                'zoom_center_y': LaunchConfiguration('zoom_center_y'),
                'save_directory': LaunchConfiguration('save_directory'),
                'capture_key_topic': '/capture_image',
                # Temporal consistency parameters
                'temporal_window_size': LaunchConfiguration('temporal_window_size'),
                'temporal_threshold': LaunchConfiguration('temporal_threshold'),
                # Hysteresis parameters
                'hysteresis_enabled': LaunchConfiguration('hysteresis_enabled'),
                'hysteresis_low_threshold': LaunchConfiguration('hysteresis_low_threshold'),
                'hysteresis_min_duration': LaunchConfiguration('hysteresis_min_duration'),
                # Depth filtering parameters
                'depth_filtering_enabled': LaunchConfiguration('depth_filtering_enabled'),
                'use_adaptive_depth': LaunchConfiguration('use_adaptive_depth'),
                'target_inspection_distance': LaunchConfiguration('target_inspection_distance'),
                'depth_tolerance': LaunchConfiguration('depth_tolerance'),
            }
        ]
    )
    
    return LaunchDescription([
        model_path_arg,
        pix2pix_model_path_arg,
        use_tiling_arg,
        use_pix2pix_arg,
        threshold_arg,
        min_crack_percent_arg,
        skip_frames_arg,
        camera_topic_arg,
        depth_topic_arg,
        zoom_enabled_arg,
        zoom_factor_arg,
        zoom_center_x_arg,
        zoom_center_y_arg,
        save_directory_arg,
        temporal_window_size_arg,
        temporal_threshold_arg,
        hysteresis_enabled_arg,
        hysteresis_low_threshold_arg,
        hysteresis_min_duration_arg,
        depth_filtering_enabled_arg,
        use_adaptive_depth_arg,
        target_inspection_distance_arg,
        depth_tolerance_arg,
        crack_detection_node
    ])
