#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Int32
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import gc
from collections import deque
import time
from datetime import datetime

import albumentations as A
from albumentations.pytorch import ToTensorV2
from ament_index_python.packages import get_package_share_directory

# Import your models
from crack_detection.unet_model import UNet
from crack_detection.train_pix2pix import Generator
from crack_detection.unet_rt_inference_tiling6 import TiledUNetInference
from crack_detection.pix2pix_rt_inference_tiling import TiledRefinedInference


class CrackDetectionNode(Node):
    """ROS2 node for crack detection with UNet and Pix2Pix support."""
    
    def __init__(self):
        super().__init__('crack_detection_node')
        
        # Declare ROS2 parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', 'models/best_model.pth'),
                ('pix2pix_model_path', 'models/pix2pix_epoch_98_best.pth'),
                ('input_size', 384),
                ('threshold', 0.5),
                ('min_crack_percent', 5.0),
                ('use_tiling', False),
                ('use_pix2pix', False),
                ('window_size', 384),
                ('subdivisions', 2),
                ('skip_frames', 1),
                ('camera_topic', '/camera/color/image_raw'),
                ('depth_topic', '/camera/depth/image'),
                ('publish_visualization', True),
                # Zoom parameters
                ('zoom_enabled', False),
                ('zoom_factor', 2.0),
                ('zoom_center_x', 0.5),
                ('zoom_center_y', 0.5),
                # Image capture parameters
                ('save_directory', 'crack_captures'),
                ('capture_key_topic', '/capture_image'),
                # Temporal consistency parameters
                ('temporal_window_size', 5),
                ('temporal_threshold', 4),
                # Hysteresis parameters
                ('hysteresis_enabled', True),
                ('hysteresis_low_threshold', 3.0),
                ('hysteresis_min_duration', 0.5),
                # Depth filtering parameters
                ('depth_filtering_enabled', True),
                ('use_adaptive_depth', True),
                ('target_inspection_distance', 1.5),
                ('depth_tolerance', 0.1),
            ]
        )
        
        # Get parameters
        model_path_param = self.get_parameter('model_path').value
        pix2pix_path_param = self.get_parameter('pix2pix_model_path').value
        
        # Convert relative paths to absolute using package share directory
        if not os.path.isabs(model_path_param):
            package_share = get_package_share_directory('crack_detection')
            self.model_path = os.path.join(package_share, model_path_param)
        else:
            self.model_path = model_path_param
            
        if not os.path.isabs(pix2pix_path_param):
            package_share = get_package_share_directory('crack_detection')
            self.pix2pix_model_path = os.path.join(package_share, pix2pix_path_param)
        else:
            self.pix2pix_model_path = pix2pix_path_param
        
        self.input_size = self.get_parameter('input_size').value
        self.threshold = self.get_parameter('threshold').value
        self.min_crack_percent = self.get_parameter('min_crack_percent').value
        self.use_tiling = self.get_parameter('use_tiling').value
        self.use_pix2pix = self.get_parameter('use_pix2pix').value
        self.window_size = self.get_parameter('window_size').value
        self.subdivisions = self.get_parameter('subdivisions').value
        self.skip_frames = self.get_parameter('skip_frames').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.publish_visualization = self.get_parameter('publish_visualization').value
        
        # Get zoom parameters
        self.zoom_enabled = self.get_parameter('zoom_enabled').value
        self.zoom_factor = self.get_parameter('zoom_factor').value
        self.zoom_center_x = self.get_parameter('zoom_center_x').value
        self.zoom_center_y = self.get_parameter('zoom_center_y').value
        
        # Get capture parameters
        self.save_directory = self.get_parameter('save_directory').value
        self.capture_key_topic = self.get_parameter('capture_key_topic').value
        
        # Get temporal consistency parameters
        self.temporal_window_size = self.get_parameter('temporal_window_size').value
        self.temporal_threshold = self.get_parameter('temporal_threshold').value
        
        # Get hysteresis parameters
        self.hysteresis_enabled = self.get_parameter('hysteresis_enabled').value
        self.hysteresis_low_threshold = self.get_parameter('hysteresis_low_threshold').value
        self.hysteresis_min_duration = self.get_parameter('hysteresis_min_duration').value
        
        # Get depth filtering parameters
        self.depth_filtering_enabled = self.get_parameter('depth_filtering_enabled').value
        self.use_adaptive_depth = self.get_parameter('use_adaptive_depth').value
        self.target_inspection_distance = self.get_parameter('target_inspection_distance').value
        self.depth_tolerance = self.get_parameter('depth_tolerance').value
        
        # Create save directory if it doesn't exist
        if not os.path.isabs(self.save_directory):
            self.save_directory = os.path.join(os.getcwd(), self.save_directory)
        os.makedirs(self.save_directory, exist_ok=True)
        
        self.get_logger().info('='*60)
        self.get_logger().info('Crack Detection Node Initializing...')
        self.get_logger().info('='*60)
        self.get_logger().info(f'UNet model path: {self.model_path}')
        if self.use_pix2pix:
            self.get_logger().info(f'Pix2Pix model path: {self.pix2pix_model_path}')
        self.get_logger().info(f'Detection threshold: {self.threshold}')
        self.get_logger().info(f'Minimum crack %: {self.min_crack_percent}%')
        self.get_logger().info(f'Frame skipping: Process every {self.skip_frames} frame(s)')
        self.get_logger().info(f'Camera topic: {self.camera_topic}')
        self.get_logger().info(f'Depth topic: {self.depth_topic}')
        
        # Log temporal consistency settings
        self.get_logger().info(f'Temporal window size: {self.temporal_window_size} frames')
        self.get_logger().info(f'Temporal threshold: {self.temporal_threshold}/{self.temporal_window_size} frames must detect')
        
        # Log hysteresis settings
        if self.hysteresis_enabled:
            self.get_logger().info(f'Hysteresis enabled: Low threshold={self.hysteresis_low_threshold}%, Min duration={self.hysteresis_min_duration}s')
        else:
            self.get_logger().info('Hysteresis disabled')
        
        # Log depth filtering settings
        if self.depth_filtering_enabled:
            if self.use_adaptive_depth:
                self.get_logger().info(f'Depth filtering: ADAPTIVE (±{self.depth_tolerance}m tolerance)')
            else:
                self.get_logger().info(f'Depth filtering: FIXED ({self.target_inspection_distance}m ±{self.depth_tolerance}m)')
        else:
            self.get_logger().info('Depth filtering disabled')
        
        # Log zoom settings
        if self.zoom_enabled:
            self.get_logger().info(f'Zoom enabled: {self.zoom_factor}x at ({self.zoom_center_x:.2f}, {self.zoom_center_y:.2f})')
        else:
            self.get_logger().info('Zoom disabled')
        
        self.get_logger().info(f'Save directory: {self.save_directory}')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Device: {self.device}')
        
        # Determine mode and initialize appropriate models
        self.determine_and_initialize_mode()
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Frame counting and caching
        self.frame_count = 0
        self.last_pred_prob = None
        self.last_binary_mask = None
        self.last_crack_percentage = 0.0
        self.last_original_frame = None  # Store original frame for saving
        
        # Depth image storage
        self.latest_depth_image = None
        self.current_wall_distance = None
        
        # Zoom region cache
        self.zoom_roi = None
        
        # FPS tracking
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        # Temporal consistency tracking
        self.detection_history = deque(maxlen=self.temporal_window_size)
        self.temporal_filter_ready = False
        
        # Hysteresis state tracking
        self.currently_in_detection = False
        self.detection_start_time = None
        
        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        
        # Subscribe to depth topic
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10
        )
        
        # Subscribe to capture trigger (can be triggered by keyboard or button)
        self.capture_sub = self.create_subscription(
            Bool,
            self.capture_key_topic,
            self.capture_callback,
            10
        )
        
        # Subscribe to AMCL pose for crack location tracking
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/a200_1103/amcl_pose',
            self.pose_callback,
            10
        )
        
        # Store current pose
        self.current_pose = None
        
        # Create publishers
        self.detection_pub = self.create_publisher(
            String,
            '/crack_detection/result',
            10
        )
        
        self.crack_detected_pub = self.create_publisher(
            Bool,
            '/crack_detection/detected',
            10
        )
        
        self.crack_center_pub = self.create_publisher(
            Point,
            '/crack_detection/center_pixel',
            10
        )
        
        # Publisher for crack location markers in RViz
        self.marker_pub = self.create_publisher(
            Marker,
            '/visualization_marker',
            10
        )
        
        # Publisher for robot pose when crack is detected
        self.robot_pose_pub = self.create_publisher(
            Point,
            '/crack_detection/robot_pose',
            10
        )
        
        if self.publish_visualization:
            self.viz_pub = self.create_publisher(
                Image,
                '/crack_detection/visualization',
                10
            )
        
        self.get_logger().info('✓ Initialization complete!')
        self.get_logger().info('⏳ Temporal filter: Collecting initial frames...')
        self.get_logger().info('Waiting for camera images...')
        self.get_logger().info('='*60)
    
    def depth_callback(self, msg):
        """Callback for depth image messages."""
        try:
            # Convert depth image to numpy array
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Check encoding and convert to meters if needed
            if msg.encoding == '16UC1':  # Depth in millimeters
                self.latest_depth_image = depth_image.astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':  # Depth already in meters
                self.latest_depth_image = depth_image.astype(np.float32)
            else:
                self.get_logger().warn(f'Unknown depth encoding: {msg.encoding}')
                self.latest_depth_image = depth_image.astype(np.float32)
                
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')
    
    def pose_callback(self, msg):
        """Callback for AMCL pose messages."""
        # Store the latest pose
        self.current_pose = msg.pose.pose.position
    
    def get_dominant_depth(self, depth_image):
        """
        Find the most common depth in the image (likely the wall surface).
        
        Args:
            depth_image: Depth values in meters
            
        Returns:
            float: Dominant depth value (most common distance)
        """
        # Flatten and filter out invalid readings (zeros and very far values)
        valid_depths = depth_image[(depth_image > 0.3) & (depth_image < 5.0)].flatten()
        
        if len(valid_depths) == 0:
            return None
        
        # Use histogram to find most common depth
        hist, bin_edges = np.histogram(valid_depths, bins=50, range=(0.3, 5.0))
        dominant_bin = np.argmax(hist)
        dominant_depth = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2.0
        
        return dominant_depth
    
    def apply_depth_filter(self, binary_mask, depth_image):
        """
        Filter detections based on depth - only keep pixels at wall surface depth.
        
        Args:
            binary_mask: Binary crack detection mask
            depth_image: Depth values in meters
            
        Returns:
            tuple: (filtered_mask, depth_mask_visualization, wall_distance)
        """
        if depth_image is None or not self.depth_filtering_enabled:
            return binary_mask, None, None
        
        # Determine target depth
        if self.use_adaptive_depth:
            # Automatically find the wall (most common depth)
            wall_depth = self.get_dominant_depth(depth_image)
            if wall_depth is None:
                self.get_logger().warn('Could not determine wall depth, skipping depth filter')
                return binary_mask, None, None
        else:
            # Use fixed target distance
            wall_depth = self.target_inspection_distance
        
        # Define valid depth range
        min_depth = wall_depth - self.depth_tolerance
        max_depth = wall_depth + self.depth_tolerance
        
        # Create depth mask (True where depth is in valid range)
        valid_depth_mask = (depth_image >= min_depth) & (depth_image <= max_depth) & (depth_image > 0)
        
        # Convert to uint8 for bitwise operations
        depth_mask_uint8 = valid_depth_mask.astype(np.uint8) * 255
        
        # Apply depth mask to crack detection mask
        filtered_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=depth_mask_uint8)
        
        return filtered_mask, depth_mask_uint8, wall_depth
    
    def apply_temporal_filter(self, current_frame_detection):
        """
        Apply temporal consistency check to reduce false positives.
        
        Args:
            current_frame_detection (bool): Detection result for current frame
            
        Returns:
            bool: Final detection decision after temporal filtering
        """
        # Add current frame detection to history
        self.detection_history.append(current_frame_detection)
        
        # Check if we have enough frames for temporal filtering
        if len(self.detection_history) < self.temporal_window_size:
            # Not enough data yet, default to no detection during startup
            if not self.temporal_filter_ready and len(self.detection_history) == self.temporal_window_size:
                self.temporal_filter_ready = True
                self.get_logger().info('✓ Temporal filter ready! Detection system now active.')
            return False
        
        # Count how many recent frames detected a crack
        detection_count = sum(self.detection_history)
        
        # Apply temporal threshold
        is_crack_detected = detection_count >= self.temporal_threshold
        
        return is_crack_detected
    
    def apply_hysteresis(self, crack_percentage, temporal_detection):
        """
        Apply hysteresis to prevent rapid on/off flickering.
        
        Args:
            crack_percentage (float): Current frame's crack percentage
            temporal_detection (bool): Result from temporal filter
            
        Returns:
            bool: Final detection decision with hysteresis applied
        """
        if not self.hysteresis_enabled:
            # Hysteresis disabled, just use temporal detection
            return temporal_detection
        
        current_time = time.time()
        
        if not self.currently_in_detection:
            # Currently NOT detecting
            # Turn ON if temporal filter says crack AND percentage above upper threshold
            if temporal_detection and crack_percentage >= self.min_crack_percent:
                self.currently_in_detection = True
                self.detection_start_time = current_time
                self.get_logger().info(f'🔵 Detection ACTIVATED (crack: {crack_percentage:.2f}%)')
                return True
            else:
                return False
        
        else:
            # Currently IN detection mode
            
            # Check minimum duration (time-based hysteresis)
            time_in_detection = current_time - self.detection_start_time
            if time_in_detection < self.hysteresis_min_duration:
                # Still within minimum duration, stay ON
                return True
            
            # Check if we should turn OFF
            # Turn OFF only if percentage drops below LOWER threshold
            if crack_percentage < self.hysteresis_low_threshold:
                self.currently_in_detection = False
                self.detection_start_time = None
                self.get_logger().info(f'🔵 Detection DEACTIVATED (crack: {crack_percentage:.2f}%)')
                return False
            else:
                # Stay ON (above lower threshold)
                return True
    
    def calculate_zoom_roi(self, frame_shape):
        """Calculate the Region of Interest for zooming."""
        height, width = frame_shape[:2]
        
        # Calculate new dimensions after zoom
        new_width = int(width / self.zoom_factor)
        new_height = int(height / self.zoom_factor)
        
        # Calculate center point in pixels
        center_x_px = int(width * self.zoom_center_x)
        center_y_px = int(height * self.zoom_center_y)
        
        # Calculate ROI boundaries
        x1 = max(0, center_x_px - new_width // 2)
        y1 = max(0, center_y_px - new_height // 2)
        x2 = min(width, x1 + new_width)
        y2 = min(height, y1 + new_height)
        
        # Adjust if ROI goes out of bounds
        if x2 - x1 < new_width:
            if x1 == 0:
                x2 = min(width, new_width)
            else:
                x1 = max(0, width - new_width)
        
        if y2 - y1 < new_height:
            if y1 == 0:
                y2 = min(height, new_height)
            else:
                y1 = max(0, height - new_height)
        
        return (x1, y1, x2, y2)
    
    def apply_zoom(self, frame):
        """Apply zoom to the frame."""
        if not self.zoom_enabled or self.zoom_factor <= 1.0:
            return frame, None
        
        # Calculate ROI if not cached or frame size changed
        if self.zoom_roi is None or self.zoom_roi[2] - self.zoom_roi[0] != int(frame.shape[1] / self.zoom_factor):
            self.zoom_roi = self.calculate_zoom_roi(frame.shape)
        
        x1, y1, x2, y2 = self.zoom_roi
        
        # Extract ROI
        zoomed_frame = frame[y1:y2, x1:x2]
        
        # Resize back to original dimensions
        zoomed_frame = cv2.resize(zoomed_frame, (frame.shape[1], frame.shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)
        
        return zoomed_frame, self.zoom_roi
    
    def map_coordinates_to_original(self, x, y, zoom_roi):
        """Map coordinates from zoomed image back to original image."""
        if zoom_roi is None:
            return x, y
        
        x1, y1, x2, y2 = zoom_roi
        roi_width = x2 - x1
        roi_height = y2 - y1
        
        # Map from zoomed (resized) coordinates to original
        original_x = int(x * roi_width / self.last_original_frame.shape[1] + x1)
        original_y = int(y * roi_height / self.last_original_frame.shape[0] + y1)
        
        return original_x, original_y
    
    def capture_callback(self, msg):
        """Callback for image capture trigger."""
        if msg.data and self.last_binary_mask is not None and self.last_original_frame is not None:
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Save original frame
            original_path = os.path.join(self.save_directory, f"frame_{timestamp}.png")
            cv2.imwrite(original_path, self.last_original_frame)
            
            # Save binary mask
            mask_path = os.path.join(self.save_directory, f"mask_{timestamp}.png")
            cv2.imwrite(mask_path, self.last_binary_mask)
            
            # Save probability map as heatmap
            if self.last_pred_prob is not None:
                heatmap = (self.last_pred_prob * 255).astype(np.uint8)
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap_path = os.path.join(self.save_directory, f"heatmap_{timestamp}.png")
                cv2.imwrite(heatmap_path, heatmap_colored)
            
            # Save depth image if available
            if self.latest_depth_image is not None:
                depth_normalized = np.clip(self.latest_depth_image / 3.0, 0, 1) * 255
                depth_path = os.path.join(self.save_directory, f"depth_{timestamp}.png")
                cv2.imwrite(depth_path, depth_normalized.astype(np.uint8))
            
            # Save info text file
            info_path = os.path.join(self.save_directory, f"info_{timestamp}.txt")
            with open(info_path, 'w') as f:
                f.write(f"Frame: {self.frame_count}\n")
                f.write(f"Crack Percentage: {self.last_crack_percentage:.2f}%\n")
                f.write(f"Wall Distance: {self.current_wall_distance:.2f}m\n" if self.current_wall_distance else "Wall Distance: N/A\n")
                f.write(f"Zoom Enabled: {self.zoom_enabled}\n")
                f.write(f"Zoom Factor: {self.zoom_factor}\n")
                f.write(f"Depth Filtering: {self.depth_filtering_enabled}\n")
                f.write(f"Mode: {self.mode}\n")
                f.write(f"Threshold: {self.threshold}\n")
            
            self.get_logger().info(f'📸 Captured images saved to {self.save_directory} with timestamp {timestamp}')
    
    def determine_and_initialize_mode(self):
        """Determine which inference mode to use and initialize models."""
        
        if self.use_tiling and self.use_pix2pix:
            # Mode 4: UNet + Pix2Pix Tiled (Best Quality)
            self.mode = "UNET_PIX2PIX_TILED"
            self.get_logger().info(f'Mode: UNet + Pix2Pix TILED (Best Quality)')
            self.get_logger().info(f'Window size: {self.window_size}x{self.window_size}')
            self.get_logger().info(f'Subdivisions: {self.subdivisions}')
            
            self.tiled_refined_inferencer = TiledRefinedInference(
                unet_path=self.model_path,
                pix2pix_path=self.pix2pix_model_path,
                device=self.device,
                window_size=self.window_size,
                subdivisions=self.subdivisions
            )
            self.model = None
            self.pix2pix = None
            self.transforms = None
            self.tiled_inferencer = None
            
        elif self.use_tiling and not self.use_pix2pix:
            # Mode 2: UNet Tiled Only
            self.mode = "UNET_TILED"
            self.get_logger().info(f'Mode: UNet TILED (High Accuracy)')
            self.get_logger().info(f'Window size: {self.window_size}x{self.window_size}')
            self.get_logger().info(f'Subdivisions: {self.subdivisions}')
            
            self.tiled_inferencer = TiledUNetInference(
                model_path=self.model_path,
                device=self.device,
                window_size=self.window_size,
                subdivisions=self.subdivisions
            )
            self.model = None
            self.pix2pix = None
            self.transforms = None
            self.tiled_refined_inferencer = None
            
        elif not self.use_tiling and self.use_pix2pix:
            # Mode 3: UNet + Pix2Pix Fast
            self.mode = "UNET_PIX2PIX_FAST"
            self.get_logger().info(f'Mode: UNet + Pix2Pix FAST (Better Quality)')
            self.get_logger().info(f'Input size: {self.input_size}x{self.input_size}')
            
            self.model, self.pix2pix = self.load_both_models()
            self.transforms = self.get_inference_transforms()
            self.tiled_inferencer = None
            self.tiled_refined_inferencer = None
            
        else:
            # Mode 1: UNet Fast Only (Default)
            self.mode = "UNET_FAST"
            self.get_logger().info(f'Mode: UNet FAST (Real-time)')
            self.get_logger().info(f'Input size: {self.input_size}x{self.input_size}')
            
            self.model = self.load_unet_model()
            self.pix2pix = None
            self.transforms = self.get_inference_transforms()
            self.tiled_inferencer = None
            self.tiled_refined_inferencer = None
    
    def load_unet_model(self):
        """Load UNet model only."""
        self.get_logger().info(f'Loading UNet from: {self.model_path}')
        
        model = UNet(
            num_classes=1,
            align_corners=False,
            use_deconv=False,
            in_channels=3
        ).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.get_logger().info('✓ UNet loaded successfully!')
        return model
    
    def load_both_models(self):
        """Load both UNet and Pix2Pix models."""
        self.get_logger().info(f'Loading UNet from: {self.model_path}')
        
        # Load UNet
        unet = UNet(
            num_classes=1,
            align_corners=False,
            use_deconv=False,
            in_channels=3
        ).to(self.device)
        
        unet_checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        unet.load_state_dict(unet_checkpoint['model_state_dict'])
        unet.eval()
        
        self.get_logger().info(f'Loading Pix2Pix from: {self.pix2pix_model_path}')
        
        # Load Pix2Pix
        pix2pix = Generator().to(self.device)
        
        pix2pix_checkpoint = torch.load(self.pix2pix_model_path, map_location=self.device, weights_only=False)
        pix2pix.load_state_dict(pix2pix_checkpoint['generator_state_dict'])
        pix2pix.eval()
        
        self.get_logger().info('✓ Both models loaded successfully!')
        return unet, pix2pix
    
    def get_inference_transforms(self):
        """Get transforms for inference."""
        return A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def predict_frame_unet_fast(self, frame_rgb):
        """Mode 1: Fast UNet prediction."""
        original_size = frame_rgb.shape[:2]
        
        # Apply transforms
        transformed = self.transforms(image=frame_rgb)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            prediction = torch.sigmoid(output)
            
            # Resize back to original size
            pred_resized = torch.nn.functional.interpolate(
                prediction,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
            
            # Convert to numpy
            pred_numpy = pred_resized.squeeze().cpu().numpy()
            
            # Apply threshold
            binary_mask = (pred_numpy > self.threshold).astype(np.uint8) * 255
        
        # Calculate crack percentage
        crack_pixels = np.sum(binary_mask > 127)
        total_pixels = binary_mask.size
        crack_percentage = (crack_pixels / total_pixels) * 100
        
        return pred_numpy, binary_mask, crack_percentage
    
    def predict_frame_unet_tiled(self, frame_rgb):
        """Mode 2: Tiled UNet prediction."""
        original_size = frame_rgb.shape[:2]
        
        # Pad image
        padded = self.tiled_inferencer._pad_img(frame_rgb)
        padded_shape = list(padded.shape[:-1]) + [1]
        
        # Process with tiling
        subdivs = self.tiled_inferencer._windowed_subdivs(padded)
        predictions = self.tiled_inferencer._recreate_from_subdivs(subdivs, padded_shape)
        
        # Unpad
        predictions = self.tiled_inferencer._unpad_img(predictions)
        
        # Crop to original size
        pred_numpy = predictions[:original_size[0], :original_size[1], 0]
        
        # Apply threshold
        binary_mask = (pred_numpy > self.threshold).astype(np.uint8) * 255
        
        # Calculate crack percentage
        crack_pixels = np.sum(binary_mask > 127)
        total_pixels = binary_mask.size
        crack_percentage = (crack_pixels / total_pixels) * 100
        
        # Cleanup
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return pred_numpy, binary_mask, crack_percentage
    
    def predict_frame_pix2pix_fast(self, frame_rgb):
        """Mode 3: Fast UNet + Pix2Pix prediction."""
        original_size = frame_rgb.shape[:2]
        
        # Apply transforms
        transformed = self.transforms(image=frame_rgb)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Stage 1: UNet prediction
            unet_output = self.model(input_tensor)
            if isinstance(unet_output, (tuple, list)):
                unet_output = unet_output[0]
            unet_pred = torch.sigmoid(unet_output)
            
            # Stage 2: Pix2Pix residual refinement
            residual = self.pix2pix(unet_pred)
            
            # Stage 3: Combine predictions
            refined_pred = torch.clamp(unet_pred + residual, 0, 1)
            
            # Resize back to original size
            pred_resized = torch.nn.functional.interpolate(
                refined_pred,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
            
            # Convert to numpy
            pred_numpy = pred_resized.squeeze().cpu().numpy()
            
            # Apply threshold
            binary_mask = (pred_numpy > self.threshold).astype(np.uint8) * 255
        
        # Calculate crack percentage
        crack_pixels = np.sum(binary_mask > 127)
        total_pixels = binary_mask.size
        crack_percentage = (crack_pixels / total_pixels) * 100
        
        return pred_numpy, binary_mask, crack_percentage
    
    def predict_frame_pix2pix_tiled(self, frame_rgb):
        """Mode 4: Tiled UNet + Pix2Pix prediction."""
        original_size = frame_rgb.shape[:2]
        
        # Pad image
        padded = self.tiled_refined_inferencer._pad_img(frame_rgb)
        padded_shape = list(padded.shape[:-1]) + [1]
        
        # Process with tiling + Pix2Pix
        subdivs = self.tiled_refined_inferencer._windowed_subdivs(padded)
        predictions = self.tiled_refined_inferencer._recreate_from_subdivs(subdivs, padded_shape)
        
        # Unpad
        predictions = self.tiled_refined_inferencer._unpad_img(predictions)
        
        # Crop to original size
        pred_numpy = predictions[:original_size[0], :original_size[1], 0]
        
        # Apply threshold
        binary_mask = (pred_numpy > self.threshold).astype(np.uint8) * 255
        
        # Calculate crack percentage
        crack_pixels = np.sum(binary_mask > 127)
        total_pixels = binary_mask.size
        crack_percentage = (crack_pixels / total_pixels) * 100
        
        # Cleanup
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return pred_numpy, binary_mask, crack_percentage
    
    def predict_frame(self, frame_rgb):
        """Route to appropriate prediction method based on mode."""
        if self.mode == "UNET_FAST":
            return self.predict_frame_unet_fast(frame_rgb)
        elif self.mode == "UNET_TILED":
            return self.predict_frame_unet_tiled(frame_rgb)
        elif self.mode == "UNET_PIX2PIX_FAST":
            return self.predict_frame_pix2pix_fast(frame_rgb)
        elif self.mode == "UNET_PIX2PIX_TILED":
            return self.predict_frame_pix2pix_tiled(frame_rgb)
    
    def calculate_crack_center(self, binary_mask):
        """Calculate the center point of detected cracks."""
        crack_pixels = np.where(binary_mask > 127)
        
        if len(crack_pixels[0]) > 0:
            center_y = int(np.mean(crack_pixels[0]))
            center_x = int(np.mean(crack_pixels[1]))
            return center_x, center_y
        else:
            return 0, 0
    
    def create_visualization(self, frame_bgr, pred_prob, binary_mask, crack_percentage, 
                           is_crack_detected, zoom_roi, temporal_status, depth_viz, wall_distance):
        """Create visualization with heatmap and overlay."""
        # Store original frame for zoom visualization
        original_frame = self.last_original_frame.copy() if self.last_original_frame is not None else frame_bgr.copy()
        
        # Draw zoom region on original frame if zoom is enabled
        if self.zoom_enabled and zoom_roi is not None:
            x1, y1, x2, y2 = zoom_roi
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(original_frame, f"Zoom: {self.zoom_factor:.1f}x", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Create heatmap
        heatmap = (pred_prob * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create crack overlay
        overlay = frame_bgr.copy()
        crack_pixels = binary_mask > 127
        overlay[crack_pixels] = [0, 0, 255]  # Red for cracks
        blended = cv2.addWeighted(frame_bgr, 0.7, overlay, 0.3, 0)
        
        
        # Create depth visualization if available
        if depth_viz is not None and self.latest_depth_image is not None:
            # Get target dimensions from RGB frame
            target_height = original_frame.shape[0]
            target_width = original_frame.shape[1]
            
            # Resize depth image to match RGB frame size
            depth_resized = cv2.resize(self.latest_depth_image,
                                    (target_width, target_height),
                                    interpolation=cv2.INTER_NEAREST)
            
            # Resize depth mask to match
            depth_viz_resized = cv2.resize(depth_viz,
                                        (target_width, target_height),
                                        interpolation=cv2.INTER_NEAREST)
            
            # Colorize depth for visualization
            depth_normalized = np.clip(depth_resized / 3.0, 0, 1) * 255
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Overlay valid depth region in green
            depth_colored[depth_viz_resized > 0] = [0, 255, 0]
            
            # Stack: original, heatmap, blended, depth (all same size now)
            combined = np.hstack([original_frame, heatmap_colored, blended, depth_colored])
        else:
            # Stack without depth: original, heatmap, blended
            combined = np.hstack([original_frame, heatmap_colored, blended])
        
        # Add info panel
        panel_height = 120
        info_panel = np.zeros((panel_height, combined.shape[1], 3), dtype=np.uint8)
        
        # FPS
        fps = self.calculate_fps()
        cv2.putText(info_panel, f"FPS: {fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Crack percentage
        cv2.putText(info_panel, f"Crack: {crack_percentage:.2f}%", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Wall distance
        if wall_distance is not None:
            cv2.putText(info_panel, f"Wall: {wall_distance:.2f}m", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Zoom status
        zoom_text = f"Zoom: {self.zoom_factor:.1f}x" if self.zoom_enabled else "Zoom: OFF"
        cv2.putText(info_panel, zoom_text, (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Temporal filter status
        cv2.putText(info_panel, f"Temporal: {temporal_status}", (250, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Hysteresis status
        hysteresis_text = "ON" if self.currently_in_detection else "OFF"
        hysteresis_color = (0, 255, 0) if self.currently_in_detection else (100, 100, 100)
        cv2.putText(info_panel, f"Hysteresis: {hysteresis_text}", (250, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hysteresis_color, 1)
        
        # Depth filter status
        depth_text = "ADAPTIVE" if self.use_adaptive_depth else f"FIXED {self.target_inspection_distance}m"
        if self.depth_filtering_enabled:
            cv2.putText(info_panel, f"Depth: {depth_text}", (250, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Mode
        cv2.putText(info_panel, f"Mode: {self.mode}", (250, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Frame count
        cv2.putText(info_panel, f"Frame: {self.frame_count}", (500, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Capture instruction
        cv2.putText(info_panel, "Publish Bool to /capture_image", (500, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        
        # Detection status
        if is_crack_detected:
            status_text = "CRACK DETECTED"
            status_color = (0, 0, 255)  # Red
        else:
            status_text = "NO CRACK"
            status_color = (0, 255, 0)  # Green
        
        text_x = combined.shape[1] - 250
        cv2.putText(info_panel, status_text, (text_x, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Combine with info panel
        result = np.vstack([info_panel, combined])
        
        return result
    
    def calculate_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        time_diff = current_time - self.last_time
        self.last_time = current_time
        
        if time_diff > 0:
            fps = 1.0 / time_diff
            self.fps_queue.append(fps)
        
        return np.mean(self.fps_queue) if len(self.fps_queue) > 0 else 0.0
    
    def image_callback(self, msg):
        """Callback function for incoming camera images."""
        self.frame_count += 1
        
        # Convert ROS Image to OpenCV
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        # Store original frame for saving and visualization
        self.last_original_frame = frame_bgr.copy()
        
        # Apply zoom if enabled
        zoomed_frame_bgr, zoom_roi = self.apply_zoom(frame_bgr)
        
        # Convert BGR to RGB for model (use zoomed frame)
        frame_rgb = cv2.cvtColor(zoomed_frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Check if we should process this frame
        if self.frame_count % self.skip_frames == 1 or self.skip_frames == 1:
            # Run inference on zoomed frame
            pred_prob, binary_mask, crack_percentage_raw = self.predict_frame(frame_rgb)
            
            # Apply depth filtering BEFORE calculating final crack percentage
            if self.depth_filtering_enabled and self.latest_depth_image is not None:
                # Resize depth image to match zoomed frame if needed
                depth_resized = cv2.resize(self.latest_depth_image, 
                                          (binary_mask.shape[1], binary_mask.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                
                binary_mask, depth_viz, wall_distance = self.apply_depth_filter(binary_mask, depth_resized)
                self.current_wall_distance = wall_distance
            else:
                depth_viz = None
                self.current_wall_distance = None
            
            # Recalculate crack percentage after depth filtering
            crack_pixels = np.sum(binary_mask > 127)
            total_pixels = binary_mask.size
            crack_percentage = (crack_pixels / total_pixels) * 100
            
            # Cache results
            self.last_pred_prob = pred_prob
            self.last_binary_mask = binary_mask
            self.last_crack_percentage = crack_percentage
        else:
            # Use cached results
            pred_prob = self.last_pred_prob
            binary_mask = self.last_binary_mask
            crack_percentage = self.last_crack_percentage
            depth_viz = None
        
        # Determine current frame detection (single frame decision)
        current_frame_detection = crack_percentage >= self.min_crack_percent
        
        # Apply temporal consistency filter
        temporal_detection = self.apply_temporal_filter(current_frame_detection)
        
        # Apply hysteresis to final detection
        is_crack_detected = self.apply_hysteresis(crack_percentage, temporal_detection)
        
        # Create temporal status string for visualization
        detection_count = sum(self.detection_history)
        temporal_status = f"{detection_count}/{len(self.detection_history)} frames"
        
        # Calculate crack center (in zoomed frame coordinates)
        center_x_zoomed, center_y_zoomed = self.calculate_crack_center(binary_mask)
        
        # Map coordinates back to original frame if zoom is enabled
        if self.zoom_enabled and zoom_roi is not None:
            center_x, center_y = self.map_coordinates_to_original(
                center_x_zoomed, center_y_zoomed, zoom_roi
            )
        else:
            center_x, center_y = center_x_zoomed, center_y_zoomed
        
        # Print detection result
        if is_crack_detected:
            zoom_info = f" [Zoom: {self.zoom_factor}x]" if self.zoom_enabled else ""
            wall_info = f" [Wall: {self.current_wall_distance:.2f}m]" if self.current_wall_distance else ""
            self.get_logger().info(
                f'🔴 CRACK DETECTED at pixel ({center_x}, {center_y}){zoom_info}{wall_info} | '
                f'Coverage: {crack_percentage:.2f}% | Temporal: {temporal_status} | Frame: {self.frame_count}'
            )
            # Log AMCL pose on separate line
            if self.current_pose is not None:
                self.get_logger().info(
                    f'📍 Robot Pose: x={self.current_pose.x:.3f}m, y={self.current_pose.y:.3f}m, z={self.current_pose.z:.3f}m'
                )
        
        # Get timestamp from message header
        timestamp_sec = msg.header.stamp.sec
        timestamp_nanosec = msg.header.stamp.nanosec
        
        # Publish detection result with all metadata
        result_msg = String()
        result_msg.data = (f'detected={is_crack_detected},crack_percent={crack_percentage:.2f},'
                          f'center_x={center_x},center_y={center_y},frame={self.frame_count},'
                          f'zoom_enabled={self.zoom_enabled},zoom_factor={self.zoom_factor},'
                          f'temporal_status={temporal_status},'
                          f'wall_distance={self.current_wall_distance if self.current_wall_distance else 0.0},'
                          f'hysteresis_active={self.currently_in_detection},'
                          f'timestamp_sec={timestamp_sec},timestamp_nanosec={timestamp_nanosec}')
        self.detection_pub.publish(result_msg)
        
        # Publish boolean detection
        detected_msg = Bool()
        detected_msg.data = bool(is_crack_detected)
        self.crack_detected_pub.publish(detected_msg)
        
        # Publish crack center (in original frame coordinates)
        if is_crack_detected:
            center_msg = Point()
            center_msg.x = float(center_x)
            center_msg.y = float(center_y)
            center_msg.z = 0.0
            self.crack_center_pub.publish(center_msg)
            
            # Publish robot pose from AMCL when crack is detected
            if self.current_pose is not None:
                pose_msg = Point()
                pose_msg.x = self.current_pose.x
                pose_msg.y = self.current_pose.y
                pose_msg.z = self.current_pose.z
                self.robot_pose_pub.publish(pose_msg)
            
            # Publish red marker at robot's current pose when crack is detected
            if self.current_pose is not None:
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = 'crack_locations'
                marker.id = self.frame_count  # Use frame count for unique IDs
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = self.current_pose.x
                marker.pose.position.y = self.current_pose.y
                marker.pose.position.z = 0.1  # Slightly above ground
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.lifetime.sec = 0  # Marker persists forever
                self.marker_pub.publish(marker)
        
        # Publish visualization
        if self.publish_visualization:
            viz_image = self.create_visualization(
                zoomed_frame_bgr, pred_prob, binary_mask, crack_percentage, 
                is_crack_detected, zoom_roi, temporal_status, depth_viz, self.current_wall_distance
            )
            
            try:
                viz_msg = self.bridge.cv2_to_imgmsg(viz_image, encoding='bgr8')
                viz_msg.header = msg.header
                self.viz_pub.publish(viz_msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish visualization: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    node = CrackDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down crack detection node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()