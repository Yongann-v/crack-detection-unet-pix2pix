"""
Tests for the Insta360 ONE R undistortion used by crack_detection_node.

Runs against the real per-lens calibrations in crack_detection/calibration_data/.

Run (from the repo root, crack-detection-unet-pix2pix/):
    python3 -m pytest Agent_md/test_undistortion_template.py -v -s     # -s prints the latency benchmark

To adopt: copy to crack_detection/test/test_undistortion.py (paths are resolved from this file).

Reference: claude.md TASK 2 / TASK 5
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest


def _find_package_root():
    """Directory containing calibration_data/ and the crack_detection Python package."""
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent / 'crack_detection', parent):
            if (candidate / 'calibration_data').is_dir() and (candidate / 'crack_detection').is_dir():
                return candidate
    raise RuntimeError("Cannot find crack_detection/calibration_data from " + __file__)


PKG_ROOT = _find_package_root()
CALIB_DIR = PKG_ROOT / 'calibration_data'
MODELS_DIR = PKG_ROOT / 'models'
sys.path.insert(0, str(PKG_ROOT))

from crack_detection.undistort import Insta360Undistorter  # noqa: E402
from crack_detection.insta360_lens import crop_lens  # noqa: E402

LENS_W, LENS_H = 1920, 540
BALANCE = 0.5   # Same as the node default


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module", params=['front', 'back'])
def calib_path(request):
    path = CALIB_DIR / f'insta360_oner_{request.param}.yaml'
    if not path.exists():
        pytest.skip(f"{path.name} not found; run calibrate_insta360 for the {request.param} lens")
    return path


@pytest.fixture(scope="module")
def undistorter(calib_path):
    return Insta360Undistorter.from_file(str(calib_path), balance=BALANCE)


@pytest.fixture
def lens_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (LENS_H, LENS_W, 3), dtype=np.uint8)


def _project_line(u, start, end, n=60):
    """Project a straight 3D segment through the calibrated (distorted) lens model."""
    pts = np.linspace(start, end, n).reshape(-1, 1, 3).astype(np.float64)
    if u.model == 'fisheye':
        img, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), u.K, u.D)
    else:
        img, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), u.K, u.D)
    return img.reshape(-1, 2)


def _to_undistorted(u, pts):
    """Map distorted pixel coordinates to the undistorter's output image (new_K)."""
    pts = pts.reshape(-1, 1, 2).astype(np.float64)
    if u.model == 'fisheye':
        out = cv2.fisheye.undistortPoints(pts, u.K, u.D, P=u.new_K)
    else:
        out = cv2.undistortPoints(pts, u.K, u.D, P=u.new_K)
    return out.reshape(-1, 2)


def _line_residual(pts):
    """Max perpendicular distance (px) of points from their best-fit line."""
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return np.abs(centred @ vt[1]).max()


# ============================================================================
# TEST SUITE 1: Calibration files
# ============================================================================

class TestCalibrationFile:

    def test_metadata(self, undistorter, calib_path):
        assert undistorter.model in ('fisheye', 'rational')
        assert undistorter.lens in calib_path.name
        assert undistorter.image_size == (LENS_W, LENS_H), "Calibrate on one 1920x540 lens strip"

    def test_reprojection_error(self, calib_path):
        fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
        rms = fs.getNode('rms_px').real()
        n = int(fs.getNode('num_images').real())
        fs.release()
        assert rms < 0.5, f"Calibration RMS {rms:.3f}px; recapture with better coverage"
        assert n >= 30, f"Only {n} images; capture 30+"

    def test_principal_point_inside_image(self, undistorter):
        cx, cy = undistorter.K[0, 2], undistorter.K[1, 2]
        assert 0.3 * LENS_W < cx < 0.7 * LENS_W
        assert 0.3 * LENS_H < cy < 0.7 * LENS_H


# ============================================================================
# TEST SUITE 2: Remap tables and output
# ============================================================================

class TestUndistortion:

    def test_remap_tables(self, undistorter):
        assert undistorter.map1.shape[:2] == (LENS_H, LENS_W)
        assert undistorter.map1.dtype == np.int16, "Expected fixed-point CV_16SC2 maps"

    def test_output_shape_and_dtype(self, undistorter, lens_image):
        out = undistorter.undistort_frame(lens_image)
        assert out.shape == lens_image.shape
        assert out.dtype == lens_image.dtype

    def test_grayscale(self, undistorter, lens_image):
        gray = cv2.cvtColor(lens_image, cv2.COLOR_BGR2GRAY)
        assert undistorter.undistort_frame(gray).shape == (LENS_H, LENS_W)

    def test_wrong_size_rejected(self, undistorter):
        stacked = np.zeros((2 * LENS_H, LENS_W, 3), np.uint8)
        with pytest.raises(ValueError):
            undistorter.undistort_frame(stacked)

    def test_valid_area(self, undistorter):
        """With balance 0.5 most of the output should come from real pixels (black arcs only)."""
        valid = undistorter.undistort_frame(np.full((LENS_H, LENS_W), 255, np.uint8)) > 0
        assert valid.mean() > 0.85, f"Only {valid.mean():.0%} valid pixels at balance {BALANCE}"

    def test_mask_stays_binary(self, undistorter):
        mask = np.zeros((LENS_H, LENS_W), np.uint8)
        cv2.line(mask, (200, 100), (1700, 450), 255, 3)
        out = undistorter.undistort_mask(mask)
        assert set(np.unique(out)) <= {0, 255}
        assert (out > 0).sum() > 0.5 * (mask > 0).sum()


# ============================================================================
# TEST SUITE 3: Straight lines stay straight
# ============================================================================

class TestStraightLines:

    # 3D segments (camera frame, metres) spanning the wide field of view
    SEGMENTS = [
        ((-2.0, -0.35, 1.0), (2.0, -0.35, 1.0)),   # horizontal, upper part of strip
        ((-2.0, 0.3, 1.0), (2.0, 0.3, 1.0)),       # horizontal, lower part
        ((-1.2, -0.4, 1.0), (-1.2, 0.4, 1.0)),     # vertical, left edge region
        ((1.3, -0.4, 1.0), (1.3, 0.4, 1.0)),       # vertical, right edge region
    ]

    @pytest.mark.parametrize('segment', SEGMENTS)
    def test_line_straightened(self, undistorter, segment):
        distorted = _project_line(undistorter, *segment)
        inside = ((distorted[:, 0] >= 0) & (distorted[:, 0] < LENS_W)
                  & (distorted[:, 1] >= 0) & (distorted[:, 1] < LENS_H))
        distorted = distorted[inside]
        if len(distorted) < 10:
            pytest.skip("Segment falls outside this lens strip")

        undistorted = _to_undistorted(undistorter, distorted)
        before, after = _line_residual(distorted), _line_residual(undistorted)
        assert after < 1.0, f"Line still bends by {after:.2f}px after undistortion"
        assert after < before, "Undistortion did not straighten the line"

    def test_remap_matches_model(self, undistorter):
        """Each output pixel must sample the input where the lens model says it came from."""
        distorted = _project_line(undistorter, (-1.0, -0.2, 1.0), (1.0, 0.25, 1.0), n=40)
        undistorted = _to_undistorted(undistorter, distorted)
        map_x, map_y = cv2.convertMaps(undistorter.map1, undistorter.map2, cv2.CV_32FC1)
        errs = []
        for (xd, yd), (xu, yu) in zip(distorted, undistorted):
            xi, yi = int(round(xu)), int(round(yu))
            if 0 <= xi < LENS_W and 0 <= yi < LENS_H:
                errs.append(np.hypot(map_x[yi, xi] - xd, map_y[yi, xi] - yd))
        assert errs, "No sample points landed inside the output image"
        assert np.max(errs) < 2.0, f"Remap disagrees with lens model by up to {np.max(errs):.2f}px"


# ============================================================================
# TEST SUITE 4: Stacked frame handling
# ============================================================================

class TestCropLens:

    def test_halves(self):
        frame = np.zeros((2 * LENS_H, LENS_W), np.uint8)
        frame[:LENS_H] = 1   # back lens on top
        frame[LENS_H:] = 2   # front lens on bottom
        assert (crop_lens(frame, 'back') == 1).all()
        assert (crop_lens(frame, 'front') == 2).all()

    def test_bad_lens(self):
        with pytest.raises(ValueError):
            crop_lens(np.zeros((2 * LENS_H, LENS_W), np.uint8), 'left')


# ============================================================================
# TEST SUITE 5: Performance
# ============================================================================

class TestLatency:

    def test_latency_per_frame(self, undistorter, lens_image):
        for _ in range(5):
            undistorter.undistort_frame(lens_image)
        times = []
        for _ in range(200):
            start = time.perf_counter()
            undistorter.undistort_frame(lens_image)
            times.append((time.perf_counter() - start) * 1000)
        mean_ms = float(np.mean(times))
        print(f"\n[BENCHMARK] {undistorter.lens}: {mean_ms:.2f} ms/frame "
              f"(p95 {np.percentile(times, 95):.2f} ms)")
        assert mean_ms < 12.0, f"Undistortion {mean_ms:.2f}ms exceeds the 12ms budget"


# ============================================================================
# TEST SUITE 6: UNet integration (needs torch + model weights)
# ============================================================================

class TestUNetIntegration:

    @pytest.fixture(autouse=True)
    def check_models_available(self):
        pytest.importorskip('torch')
        if not (MODELS_DIR / 'best_model.pth').exists():
            pytest.skip(f"UNet weights not found in {MODELS_DIR}")

    @staticmethod
    def _unet_input(frame_bgr, size=384):
        """Same preprocessing as crack_detection_node (resize + ImageNet normalize)."""
        import torch
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (size, size)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)

    def test_unet_inference_on_undistorted(self, undistorter, lens_image):
        import torch
        from crack_detection.unet_model import UNet

        model = UNet(num_classes=1, align_corners=False, use_deconv=False, in_channels=3)
        ckpt = torch.load(MODELS_DIR / 'best_model.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        undistorted = undistorter.undistort_frame(lens_image)
        with torch.no_grad():
            out = model(self._unet_input(undistorted))
            if isinstance(out, (tuple, list)):
                out = out[0]
            prob = torch.nn.functional.interpolate(
                torch.sigmoid(out), size=undistorted.shape[:2], mode='bilinear', align_corners=False)

        mask = (prob.squeeze().numpy() > 0.5).astype(np.uint8) * 255
        assert mask.shape == undistorted.shape[:2]
        assert set(np.unique(mask)) <= {0, 255}


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
