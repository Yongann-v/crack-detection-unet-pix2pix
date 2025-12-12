import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal.windows import triang
from tqdm import tqdm
import gc
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your UNet model
from crack_detection.unet_model import UNet

class TiledUNetInference:
    """UNet inference with smooth tiling for memory-efficient processing."""
    
    def __init__(self, model_path, device=None, window_size=384, subdivisions=2):
        """
        Initialize tiled inference.
        
        Args:
            model_path: Path to trained model
            device: torch device (auto-detect if None)
            window_size: Size of tiles (smaller = less memory)
            subdivisions: Overlap factor (2 = 50% overlap, 4 = 75% overlap)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.subdivisions = subdivisions
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Setup transforms
        self.transforms = self.get_inference_transforms(window_size)
        
        # Cache for window functions
        self.cached_2d_windows = {}
        
        print(f"Initialized tiled inference:")
        print(f"  Device: {self.device}")
        print(f"  Window size: {window_size}")
        print(f"  Subdivisions: {subdivisions} (overlap: {100*(1-1/subdivisions):.0f}%)")
    
    def load_model(self, model_path):
        """Load trained UNet model."""
        print(f"Loading model from: {model_path}")
        
        model = UNet(
            num_classes=1,
            align_corners=False,
            use_deconv=False,
            in_channels=3
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("Model loaded successfully!")
        return model
    
    def get_inference_transforms(self, input_size):
        """Get transforms for inference."""
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _spline_window(self, window_size, power=2):
        """Create spline window function for smooth blending."""
        intersection = int(window_size/4)
        wind_outer = (abs(2*(triang(window_size))) ** power)/2
        wind_outer[intersection:-intersection] = 0
        
        wind_inner = 1 - (abs(2*(triang(window_size) - 1)) ** power)/2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0
        
        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)
        return wind
    
    def _window_2D(self, window_size, power=2):
        """Create 2D window function."""
        key = f"{window_size}_{power}"
        if key in self.cached_2d_windows:
            return self.cached_2d_windows[key]
        
        wind = self._spline_window(window_size, power)
        # Create 2D window for single channel
        wind_2d = np.outer(wind, wind)
        wind_2d = np.expand_dims(wind_2d, axis=-1)
        
        self.cached_2d_windows[key] = wind_2d
        return wind_2d
    
    def _pad_img(self, img):
        """Add reflective padding for valid tiling."""
        aug = int(round(self.window_size * (1 - 1.0/self.subdivisions)))
        more_borders = ((aug, aug), (aug, aug), (0, 0))
        return np.pad(img, pad_width=more_borders, mode='reflect')
    
    def _unpad_img(self, padded_img):
        """Remove padding."""
        aug = int(round(self.window_size * (1 - 1.0/self.subdivisions)))
        return padded_img[aug:-aug, aug:-aug]
    
    def predict_patches(self, patches):
        """
        Predict on a batch of patches.
        
        Args:
            patches: numpy array of shape (n_patches, height, width, channels)
        
        Returns:
            predictions: numpy array of shape (n_patches, height, width, 1)
        """
        predictions = []
        
        # Process in small batches to control memory
        batch_size = 4  # Adjust based on your laptop's memory
        
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            batch_tensors = []
            
            # Prepare batch tensors
            for patch in batch:
                # Apply transforms (no resize, patches are already correct size)
                transformed = self.transforms(image=patch)
                batch_tensors.append(transformed['image'])
            
            # Stack into batch
            batch_input = torch.stack(batch_tensors).to(self.device)
            
            # Predict on entire batch
            with torch.no_grad():
                output = self.model(batch_input)
                
                # Handle model output (tuple or tensor)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                
                predictions_batch = torch.sigmoid(output)
                
                # Convert to numpy
                for pred in predictions_batch:
                    pred_numpy = pred.squeeze().cpu().numpy()
                    # Add channel dimension
                    pred_numpy = np.expand_dims(pred_numpy, axis=-1)
                    predictions.append(pred_numpy)
            
            # Clear GPU cache after each batch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Garbage collection after all batches
        gc.collect()
        
        return np.array(predictions)
    
    def _windowed_subdivs(self, padded_img):
        """Create and process tiled overlapping patches."""
        window_2d = self._window_2D(self.window_size, power=2)
        step = int(self.window_size / self.subdivisions)
        padx_len = padded_img.shape[0]
        pady_len = padded_img.shape[1]
        
        subdivs = []
        
        # Extract patches
        for i in range(0, padx_len - self.window_size + 1, step):
            subdivs.append([])
            for j in range(0, pady_len - self.window_size + 1, step):
                patch = padded_img[i:i+self.window_size, j:j+self.window_size, :]
                subdivs[-1].append(patch)
        
        gc.collect()
        
        # Reshape for batch processing
        subdivs = np.array(subdivs)
        a, b, c, d, e = subdivs.shape
        subdivs = subdivs.reshape(a * b, c, d, e)
        
        # Process patches through model
        print(f"Processing {len(subdivs)} patches...")
        predictions = self.predict_patches(subdivs)
        gc.collect()
        
        # Apply window function for smooth blending
        predictions = np.array([pred * window_2d for pred in predictions])
        gc.collect()
        
        # Reshape back to grid
        predictions = predictions.reshape(a, b, c, d, 1)
        
        return predictions
    
    def _recreate_from_subdivs(self, subdivs, padded_shape):
        """Merge patches back into full image."""
        step = int(self.window_size / self.subdivisions)
        padx_len = padded_shape[0]
        pady_len = padded_shape[1]
        
        y = np.zeros(padded_shape)
        
        a = 0
        for i in range(0, padx_len - self.window_size + 1, step):
            b = 0
            for j in range(0, pady_len - self.window_size + 1, step):
                windowed_patch = subdivs[a, b]
                y[i:i+self.window_size, j:j+self.window_size] += windowed_patch
                b += 1
            a += 1
        
        return y / (self.subdivisions ** 2)
    
    def predict_image(self, image_path, threshold=0.5, use_augmentation=False):
        """
        Predict cracks in image using smooth tiling.
        
        Args:
            image_path: Path to input image
            threshold: Threshold for binary mask
            use_augmentation: Whether to use rotation/mirror augmentation (uses more memory)
        
        Returns:
            original: Original image
            pred_prob: Probability map
            binary_mask: Binary mask
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image_rgb.shape[:2]
        
        print(f"Processing image: {original_size[0]}x{original_size[1]} pixels")
        
        # Pad image
        padded = self._pad_img(image_rgb)
        padded_shape = list(padded.shape[:-1]) + [1]  # Shape for output
        
        if use_augmentation:
            # Process with rotation augmentation (8x more memory)
            predictions = self._process_with_augmentation(padded, padded_shape)
        else:
            # Process without augmentation (memory efficient)
            print("Processing without augmentation (memory efficient mode)")
            subdivs = self._windowed_subdivs(padded)
            predictions = self._recreate_from_subdivs(subdivs, padded_shape)
        
        # Unpad
        predictions = self._unpad_img(predictions)
        
        # Crop to original size and remove channel dimension
        predictions = predictions[:original_size[0], :original_size[1], 0]
        
        # Apply threshold
        binary_mask = (predictions > threshold).astype(np.uint8) * 255
        
        gc.collect()
        
        return image_rgb, predictions, binary_mask
    
    def _process_with_augmentation(self, padded, padded_shape):
        """Process with rotation/mirror augmentation (optional, uses more memory)."""
        print("Processing with rotation augmentation (8 variants)...")
        
        # Create rotations/mirrors
        pads = self._rotate_mirror_do(padded)
        results = []
        
        for idx, pad in enumerate(tqdm(pads, desc="Processing augmented versions")):
            subdivs = self._windowed_subdivs(pad)
            one_result = self._recreate_from_subdivs(subdivs, padded_shape)
            results.append(one_result)
            gc.collect()
        
        # Merge results
        return self._rotate_mirror_undo(results)
    
    def _rotate_mirror_do(self, im):
        """Create 8 rotations/mirrors."""
        mirrs = []
        mirrs.append(np.array(im))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
        im = np.array(im)[:, ::-1]
        mirrs.append(np.array(im))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
        return mirrs
    
    def _rotate_mirror_undo(self, im_mirrs):
        """Merge rotations/mirrors back."""
        origs = []
        origs.append(np.array(im_mirrs[0]))
        origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
        origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
        origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
        origs.append(np.array(im_mirrs[4])[:, ::-1])
        origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
        origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
        origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
        return np.mean(origs, axis=0)


def visualize_tiled_results(original, pred_prob, binary_mask, save_path=None):
    """Visualize results with tiling information."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Probability heatmap
    im1 = axes[1].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Crack Probability (Smooth Tiled)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Binary mask
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title('Detected Cracks', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Overlay
    overlay = original.copy()
    crack_pixels = binary_mask > 127
    overlay[crack_pixels] = [255, 0, 0]
    
    blended = cv2.addWeighted(original.astype(np.uint8), 0.7,
                             overlay.astype(np.uint8), 0.3, 0)
    
    axes[3].imshow(blended)
    axes[3].set_title('Overlay (Red = Cracks)', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_tiled_inference(model_path, image_path, output_path=None, 
                        window_size=384, subdivisions=2, threshold=0.5,
                        use_augmentation=False):
    """
    Test crack detection with tiled inference.
    
    Args:
        model_path: Path to trained model
        image_path: Path to test image
        output_path: Path to save results
        window_size: Tile size (smaller = less memory)
        subdivisions: Overlap factor (2=50%, 4=75% overlap)
        threshold: Binary threshold
        use_augmentation: Use rotation augmentation (8x memory)
    """
    
    # Initialize tiled inference
    inferencer = TiledUNetInference(
        model_path=model_path,
        window_size=window_size,
        subdivisions=subdivisions
    )
    
    # Process image
    print(f"\nProcessing: {image_path}")
    original, pred_prob, binary_mask = inferencer.predict_image(
        image_path, 
        threshold=threshold,
        use_augmentation=use_augmentation
    )
    
    # Calculate statistics
    crack_pixels = np.sum(binary_mask > 127)
    total_pixels = binary_mask.size
    crack_percentage = (crack_pixels / total_pixels) * 100
    
    print(f"\nResults:")
    print(f"  Crack coverage: {crack_percentage:.2f}%")
    print(f"  Crack pixels: {crack_pixels:,} / {total_pixels:,}")
    
    # Save visualization
    if output_path is None:
        output_path = f"tiled_crack_detection_2_{os.path.basename(image_path)}.png"
    
    visualize_tiled_results(original, pred_prob, binary_mask, output_path)
    
    return crack_percentage


# Example usage
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/best_model.pth"
    IMAGE_PATH = "new_test_images2/2025-09-23-164845.jpg"
    
    # Memory-efficient settings for laptop
    WINDOW_SIZE = 384  # Reduce to 256 if still running out of memory
    SUBDIVISIONS = 2   # Increase to 4 for more overlap but smaller steps
    USE_AUGMENTATION = False  # Set to True only if you have enough memory
    
    try:
        crack_percentage = test_tiled_inference(
            model_path=MODEL_PATH,
            image_path=IMAGE_PATH,
            window_size=WINDOW_SIZE,
            subdivisions=SUBDIVISIONS,
            use_augmentation=USE_AUGMENTATION,
            threshold=0.5
        )
        
        print(f"\n✓ Analysis complete!")
        print(f"  Found {crack_percentage:.2f}% crack coverage")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Reduce window_size (try 256 or 192)")
        print("2. Increase subdivisions for smaller steps")
        print("3. Ensure use_augmentation=False for memory efficiency")
        print("4. Check that model file and image exist")
