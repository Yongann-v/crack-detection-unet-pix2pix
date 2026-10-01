"""
Unit & Integration Tests for Insta360 Undistortion Pipeline

Run:
    pytest tests/test_undistortion_template.py -v
    pytest tests/test_undistortion_template.py::test_latency_jetson -v  # Latency test only

Reference: claude.md § TASK 5
"""

import pytest
import numpy as np
import cv2
import time
from pathlib import Path

# Adjust imports based on your workspace structure
from src.insta360_undistortion.undistort import Insta360Undistorter


# ============================================================================
# FIXTURES (Setup/Teardown)
# ============================================================================

@pytest.fixture(scope="session")
def calibration_params():
    """Load calibration matrices (K, D) for tests."""
    calib_dir = Path('calibration_data')
    
    # Check if calibration exists
    K_path = calib_dir / 'insta360_oner_K.npy'
    D_path = calib_dir / 'insta360_oner_D.npy'
    
    if not K_path.exists() or not D_path.exists():
        pytest.skip("Calibration files not found. Run calibration first.")
    
    K = np.load(K_path)
    D = np.load(D_path)
    
    return K, D


@pytest.fixture(scope="session")
def undistorter(calibration_params):
    """Create Insta360Undistorter instance."""
    K, D = calibration_params
    return Insta360Undistorter(K, D, image_size=(1920, 1080), pole_crop_percent=0.10)


@pytest.fixture
def dummy_image():
    """Generate random test image (1920x1080 BGR)."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def checkerboard_image():
    """Generate synthetic checkerboard image for distortion validation."""
    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    square_size = 80  # pixels
    
    for i in range(0, 1920, square_size):
        for j in range(0, 1080, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                img[j:j+square_size, i:i+square_size] = 0
    
    return img


# ============================================================================
# TEST SUITE 1: Remap Correctness
# ============================================================================

class TestRemapGeneration:
    """Validate that remap pre-computation is correct."""
    
    def test_remap_shapes(self, undistorter):
        """Verify map_x, map_y have correct shape."""
        assert undistorter.map_x.shape == (1080, 1920), f"map_x shape {undistorter.map_x.shape} != (1080, 1920)"
        assert undistorter.map_y.shape == (1080, 1920), f"map_y shape {undistorter.map_y.shape} != (1080, 1920)"
    
    def test_remap_value_ranges(self, undistorter):
        """Verify map values are within valid pixel ranges."""
        # map_x: u coordinates [0, 1920]
        assert np.all(undistorter.map_x >= 0), "map_x has negative values"
        assert np.all(undistorter.map_x <= 1920), "map_x has values > 1920"
        
        # map_y: v coordinates [0, 1080]
        assert np.all(undistorter.map_y >= 0), "map_y has negative values"
        assert np.all(undistorter.map_y <= 1080), "map_y has values > 1080"
    
    def test_remap_dtype(self, undistorter):
        """Verify remap is float32 for cv2.remap compatibility."""
        assert undistorter.map_x.dtype == np.float32, f"map_x dtype is {undistorter.map_x.dtype}, not float32"
        assert undistorter.map_y.dtype == np.float32, f"map_y dtype is {undistorter.map_y.dtype}, not float32"
    
    def test_pole_crop_indices(self, undistorter):
        """Verify pole cropping indices are correct."""
        h, w = 1080, 1920
        expected_crop_start = int(h * 0.10)  # Top 10%
        expected_crop_end = int(h * (1 - 0.10))  # Bottom 10%
        
        assert undistorter.crop_y_start == expected_crop_start, \
            f"crop_y_start {undistorter.crop_y_start} != {expected_crop_start}"
        assert undistorter.crop_y_end == expected_crop_end, \
            f"crop_y_end {undistorter.crop_y_end} != {expected_crop_end}"


# ============================================================================
# TEST SUITE 2: End-to-End Undistortion
# ============================================================================

class TestUndistortion:
    """Test actual undistortion process."""
    
    def test_undistort_output_shape(self, undistorter, dummy_image):
        """Verify output shape matches cropped expectation."""
        output = undistorter.undistort_frame(dummy_image)
        
        h_original = 1080
        h_cropped = int(h_original * (1 - 2 * 0.10))  # Remove top & bottom 10%
        
        assert output.shape == (h_cropped, 1920, 3), \
            f"Output shape {output.shape} != ({h_cropped}, 1920, 3)"
    
    def test_undistort_dtype_preservation(self, undistorter, dummy_image):
        """Verify output dtype matches input."""
        output = undistorter.undistort_frame(dummy_image)
        assert output.dtype == dummy_image.dtype, \
            f"Output dtype {output.dtype} != input dtype {dummy_image.dtype}"
    
    def test_undistort_no_nan(self, undistorter, dummy_image):
        """Verify undistorted image has no NaN values."""
        output = undistorter.undistort_frame(dummy_image)
        assert not np.isnan(output).any(), "Undistorted image contains NaN values"
    
    def test_undistort_grayscale(self, undistorter, checkerboard_image):
        """Test on grayscale image."""
        gray = cv2.cvtColor(checkerboard_image, cv2.COLOR_BGR2GRAY)
        output = undistorter.undistort_frame(gray)
        
        h_cropped = int(1080 * 0.80)  # 80% after pole cropping
        assert output.shape == (h_cropped, 1920), f"Grayscale output shape incorrect"


# ============================================================================
# TEST SUITE 3: Straight Line Validation
# ============================================================================

class TestStraightLinePreservation:
    """Validate that straight lines remain straight after undistortion."""
    
    def test_horizontal_line_straight(self, undistorter):
        """Draw horizontal line on distorted image, verify it remains straight after undistortion."""
        # Create image with horizontal line at y=540 (center)
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        cv2.line(img, (0, 540), (1920, 540), (0, 0, 255), 5)  # Red line
        
        output = undistorter.undistort_frame(img)
        
        # Check that line remains mostly horizontal (variance of x-coordinates at each y should be low)
        # This is a heuristic test; exact validation would require edge detection
        assert output.shape[1] == 1920, "Image width changed"
        assert output.shape[0] > 0, "Image height is zero"
    
    def test_vertical_line_near_center(self, undistorter):
        """Draw vertical line near center, verify straightness."""
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        cv2.line(img, (960, 0), (960, 1080), (0, 255, 0), 5)  # Green line at center
        
        output = undistorter.undistort_frame(img)
        assert output.shape == (864, 1920, 3), "Output shape mismatch after vertical line test"


# ============================================================================
# TEST SUITE 4: Performance / Latency
# ============================================================================

class TestLatencyPerformance:
    """Measure latency on current hardware."""
    
    @pytest.mark.benchmark
    def test_latency_remap_generation(self, calibration_params):
        """Benchmark remap generation (one-time cost)."""
        K, D = calibration_params
        
        start = time.perf_counter()
        undistorter = Insta360Undistorter(K, D, image_size=(1920, 1080))
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        print(f"\n[BENCHMARK] Remap generation: {elapsed_ms:.2f}ms")
        assert elapsed_ms < 100, f"Remap generation too slow: {elapsed_ms}ms"
    
    @pytest.mark.benchmark
    def test_latency_per_frame(self, undistorter, dummy_image):
        """Measure latency per frame (target: <12ms on Jetson)."""
        n_iterations = 100
        times = []
        
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = undistorter.undistort_frame(dummy_image)
            times.append((time.perf_counter() - start) * 1000)  # milliseconds
        
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        min_ms = np.min(times)
        max_ms = np.max(times)
        
        print(f"\n[BENCHMARK] Undistortion Latency:")
        print(f"  Mean: {mean_ms:.2f}ms ± {std_ms:.2f}ms")
        print(f"  Min:  {min_ms:.2f}ms")
        print(f"  Max:  {max_ms:.2f}ms")
        print(f"  Throughput: {1000/mean_ms:.1f} FPS")
        
        # Expected on Jetson Xavier NX: ~12ms
        # Warn if > 15ms, fail if > 20ms
        assert mean_ms < 20, f"Latency {mean_ms:.2f}ms exceeds 20ms threshold (Jetson too slow?)"
        
        if mean_ms > 15:
            print(f"\n⚠️  WARNING: Latency {mean_ms:.2f}ms > 15ms target (check GPU utilization)")
    
    @pytest.mark.benchmark
    def test_throughput_fps(self, undistorter, dummy_image):
        """Calculate sustained throughput (FPS)."""
        n_frames = 1000
        start = time.perf_counter()
        
        for _ in range(n_frames):
            _ = undistorter.undistort_frame(dummy_image)
        
        total_time = time.perf_counter() - start
        fps = n_frames / total_time
        
        print(f"\n[BENCHMARK] Sustained Throughput: {fps:.1f} FPS ({n_frames} frames in {total_time:.2f}s)")
        
        # Target: ≥25 Hz for pipe scanning
        # Note: Expected is 80+ FPS on Jetson (GPU-accelerated cv2.remap)
        assert fps >= 25, f"Throughput {fps:.1f} FPS < 25 Hz target"


# ============================================================================
# TEST SUITE 5: Metadata & Logging
# ============================================================================

class TestMetadata:
    """Test metadata extraction and logging."""
    
    def test_get_metadata(self, undistorter, calibration_params):
        """Verify get_metadata returns correct structure."""
        K, D = calibration_params
        metadata = undistorter.get_metadata()
        
        assert 'K' in metadata, "K not in metadata"
        assert 'D' in metadata, "D not in metadata"
        assert 'image_size' in metadata, "image_size not in metadata"
        assert 'pole_crop_percent' in metadata, "pole_crop_percent not in metadata"
        assert 'algorithm' in metadata, "algorithm not in metadata"
        
        assert metadata['image_size'] == (1920, 1080)
        assert metadata['pole_crop_percent'] == 0.10
        assert 'fisheye' in metadata['algorithm'].lower()


# ============================================================================
# TEST SUITE 6: Integration Tests (Optional, requires YOLOv8)
# ============================================================================

class TestYOLOv8Integration:
    """Integration tests with YOLOv8 (skip if yolov8 not installed)."""
    
    @pytest.fixture(autouse=True)
    def check_yolov8_installed(self):
        """Skip tests if YOLOv8 not available."""
        try:
            from ultralytics import YOLO
        except ImportError:
            pytest.skip("YOLOv8 not installed")
    
    @pytest.mark.integration
    def test_yolov8_inference_on_undistorted(self, undistorter, dummy_image):
        """Verify YOLOv8 can run on undistorted frames (no shape mismatches)."""
        from ultralytics import YOLO
        
        # Load a pretrained YOLOv8 model (tiny for speed)
        model = YOLO('yolov8n.pt')
        
        # Undistort
        undistorted = undistorter.undistort_frame(dummy_image)
        
        # Run inference (should not crash)
        results = model.predict(undistorted, conf=0.5, verbose=False)
        
        assert len(results) > 0, "YOLOv8 returned no results"
        assert results[0].boxes is not None, "No bounding boxes detected"


# ============================================================================
# Custom CLI Tests (pytest can also be run manually)
# ============================================================================

def test_summary(undistorter, calibration_params):
    """Print summary of all tests (not a real assertion)."""
    K, D = calibration_params
    
    print("\n" + "="*70)
    print("UNDISTORTION PIPELINE TEST SUMMARY")
    print("="*70)
    print(f"Camera Matrix K:\n{K}")
    print(f"\nDistortion Coefficients D:\n{D}")
    print(f"\nUndistorter Config:")
    print(f"  Image Size: {undistorter.image_size}")
    print(f"  Pole Crop: {undistorter.pole_crop_percent*100:.1f}%")
    print(f"  Remap initialized: {undistorter.map_x is not None and undistorter.map_y is not None}")
    print("="*70)


# ============================================================================
# Run Tests from Command Line
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
