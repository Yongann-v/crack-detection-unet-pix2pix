import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import argparse
import gc
from scipy.signal.windows import triang

# Import models
from crack_detection.unet_model import UNet
from crack_detection.train_pix2pix import Generator


class TiledRefinedInference:
    """Tiled inference for U-Net + Pix2Pix refinement with memory efficiency."""
    
    def __init__(self, unet_path, pix2pix_path, device=None, window_size=384, subdivisions=2):
        """
        Initialize tiled refined inference.
        
        Args:
            unet_path: Path to trained U-Net model
            pix2pix_path: Path to trained Pix2Pix model
            device: torch device (auto-detect if None)
            window_size: Size of tiles (smaller = less memory)
            subdivisions: Overlap factor (2 = 50% overlap, 4 = 75% overlap)
        """
        # Ensure device is a torch.device object
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.window_size = window_size
        self.subdivisions = subdivisions
        
        # Load both models
        self.unet, self.pix2pix = self.load_models(unet_path, pix2pix_path)
        
        # Setup transforms
        self.transforms = self.get_inference_transforms()
        
        # Cache for window functions
        self.cached_2d_windows = {}
        
        print(f"Initialized tiled refined inference:")
        print(f"  Device: {self.device}")
        print(f"  Window size: {window_size}")
        print(f"  Subdivisions: {subdivisions} (overlap: {100*(1-1/subdivisions):.0f}%)")
    
    def load_models(self, unet_path, pix2pix_path):
        """Load both U-Net and Pix2Pix models."""
        print(f"Loading U-Net from: {unet_path}")
        
        # Load U-Net
        unet = UNet(
            num_classes=1,
            align_corners=False,
            use_deconv=False,
            in_channels=3
        ).to(self.device)
        
        unet_checkpoint = torch.load(unet_path, map_location=self.device, weights_only=False)
        unet.load_state_dict(unet_checkpoint['model_state_dict'])
        unet.eval()
        
        print(f"Loading Pix2Pix from: {pix2pix_path}")
        
        # Load Pix2Pix Generator
        pix2pix = Generator().to(self.device)
        
        pix2pix_checkpoint = torch.load(pix2pix_path, map_location=self.device, weights_only=False)
        pix2pix.load_state_dict(pix2pix_checkpoint['generator_state_dict'])
        pix2pix.eval()
        
        print("Both models loaded successfully!")
        return unet, pix2pix
    
    def get_inference_transforms(self):
        """Get transforms for inference (no resize)."""
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
    
    def predict_patches_refined(self, patches):
        """
        Predict on patches using U-Net + Pix2Pix refinement.
        
        Args:
            patches: numpy array of shape (n_patches, height, width, channels)
        
        Returns:
            predictions: numpy array of shape (n_patches, height, width, 1)
        """
        predictions = []
        
        # Process in small batches
        batch_size = 4
        
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            batch_tensors = []
            
            # Prepare batch
            for patch in batch:
                transformed = self.transforms(image=patch)
                batch_tensors.append(transformed['image'])
            
            batch_input = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                # Stage 1: U-Net prediction
                unet_output = self.unet(batch_input)
                if isinstance(unet_output, (tuple, list)):
                    unet_output = unet_output[0]
                unet_pred = torch.sigmoid(unet_output)
                
                # Stage 2: Pix2Pix refinement
                residual = self.pix2pix(unet_pred)
                
                # Stage 3: Combine
                refined_pred = torch.clamp(unet_pred + residual, 0, 1)
                
                # Convert to numpy
                for pred in refined_pred:
                    pred_numpy = pred.squeeze().cpu().numpy()
                    pred_numpy = np.expand_dims(pred_numpy, axis=-1)
                    predictions.append(pred_numpy)
            
            # Clear GPU cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
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
        
        # Process through U-Net + Pix2Pix
        print(f"Processing {len(subdivs)} patches through U-Net + Pix2Pix...")
        predictions = self.predict_patches_refined(subdivs)
        gc.collect()
        
        # Apply window function
        predictions = np.array([pred * window_2d for pred in predictions])
        gc.collect()
        
        # Reshape back
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
    
    def predict_image(self, image_path, threshold=0.5):
        """
        Predict cracks using U-Net + Pix2Pix with tiling.
        
        Args:
            image_path: Path to input image
            threshold: Threshold for binary mask
        
        Returns:
            original: Original image
            pred_prob: Probability map
            binary_mask: Binary mask
        """
        # Load image with EXIF handling
        image_pil = Image.open(image_path)
        image_pil = ImageOps.exif_transpose(image_pil)
        image_rgb = np.array(image_pil)
        original_size = image_rgb.shape[:2]
        
        print(f"Processing image: {original_size[0]}x{original_size[1]} pixels")
        
        # Pad image
        padded = self._pad_img(image_rgb)
        padded_shape = list(padded.shape[:-1]) + [1]
        
        # Process with tiling
        subdivs = self._windowed_subdivs(padded)
        predictions = self._recreate_from_subdivs(subdivs, padded_shape)
        
        # Unpad and crop
        predictions = self._unpad_img(predictions)
        predictions = predictions[:original_size[0], :original_size[1], 0]
        
        # Apply threshold
        binary_mask = (predictions > threshold).astype(np.uint8) * 255
        
        gc.collect()
        
        return image_rgb, predictions, binary_mask


def visualize_results(original, pred_prob, binary_mask, save_path=None):
    """Create visualization of refined tiled results."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Probability heatmap
    im1 = axes[1].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Crack Probability\n(U-Net + Pix2Pix, Tiled)', fontsize=12, fontweight='bold')
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
    else:
        plt.show()
    
    plt.close()


def process_folder(inferencer, input_folder, output_folder, threshold=0.5):
    """Process all images in a folder."""
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_folder) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Results will be saved to: {output_folder}")
    print("="*60)
    
    successful = 0
    failed = 0
    crack_stats = []
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_file}")
            image_path = os.path.join(input_folder, image_file)
            
            # Predict
            original, pred_prob, binary_mask = inferencer.predict_image(
                image_path, threshold
            )
            
            # Calculate statistics
            crack_pixels = np.sum(binary_mask > 127)
            total_pixels = binary_mask.size
            crack_percentage = (crack_pixels / total_pixels) * 100
            
            crack_stats.append({
                'filename': image_file,
                'crack_percentage': crack_percentage
            })
            
            print(f"Crack coverage: {crack_percentage:.2f}%")
            
            # Save results
            output_filename = f"refined_tiled_{os.path.splitext(image_file)[0]}.png"
            output_path = os.path.join(output_folder, output_filename)
            visualize_results(original, pred_prob, binary_mask, output_path)
            
            successful += 1
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            failed += 1
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Successfully processed: {successful}/{len(image_files)} images")
    if failed > 0:
        print(f"Failed: {failed}/{len(image_files)} images")
    
    if crack_stats:
        avg_crack = np.mean([s['crack_percentage'] for s in crack_stats])
        max_crack = max(crack_stats, key=lambda x: x['crack_percentage'])
        min_crack = min(crack_stats, key=lambda x: x['crack_percentage'])
        
        print(f"\nCrack Coverage Statistics:")
        print(f"  Average: {avg_crack:.2f}%")
        print(f"  Maximum: {max_crack['crack_percentage']:.2f}% ({max_crack['filename']})")
        print(f"  Minimum: {min_crack['crack_percentage']:.2f}% ({min_crack['filename']})")
    
    print(f"\nResults saved in: {output_folder}")
    print("="*60)


def process_single_image(inferencer, image_path, output_path, threshold=0.5):
    """Process a single image."""
    
    print(f"Processing: {image_path}")
    
    # Predict
    original, pred_prob, binary_mask = inferencer.predict_image(
        image_path, threshold
    )
    
    # Calculate statistics
    crack_pixels = np.sum(binary_mask > 127)
    total_pixels = binary_mask.size
    crack_percentage = (crack_pixels / total_pixels) * 100
    
    print(f"Crack coverage: {crack_percentage:.2f}% of image")
    
    # Visualize
    if output_path is None:
        output_path = f"refined_tiled_{os.path.basename(image_path)}.png"
    
    visualize_results(original, pred_prob, binary_mask, output_path)
    print(f"Results saved to: {output_path}")
    
    return crack_percentage


def main():
    parser = argparse.ArgumentParser(
        description='Tiled crack detection using U-Net + Pix2Pix refinement'
    )
    parser.add_argument('--unet_model', type=str, default='models/best_model.pth',
                       help='Path to trained UNet model')
    parser.add_argument('--pix2pix_model', type=str, default='pix2pix_models/pix2pix_epoch_100_best.pth',
                       help='Path to trained Pix2Pix model')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image file or folder')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file or folder')
    parser.add_argument('--window_size', type=int, default=384,
                       help='Tile size (smaller = less memory, default: 384)')
    parser.add_argument('--subdivisions', type=int, default=2,
                       help='Overlap factor (2=50%%, 4=75%% overlap, default: 2)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary classification (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Initialize tiled inference
    inferencer = TiledRefinedInference(
        unet_path=args.unet_model,
        pix2pix_path=args.pix2pix_model,
        device=args.device,
        window_size=args.window_size,
        subdivisions=args.subdivisions
    )
    
    # Process input
    if os.path.isfile(args.input):
        process_single_image(inferencer, args.input, args.output, args.threshold)
    elif os.path.isdir(args.input):
        process_folder(inferencer, args.input, args.output, args.threshold)
    else:
        print(f"Error: {args.input} is not a valid file or folder")


if __name__ == "__main__":
    main()
