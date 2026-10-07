"""Real-time undistortion for one Insta360 ONE R lens.

The remap tables are built once from the calibration, so each frame costs a single cv2.remap.
Supports the OpenCV fisheye model and the pinhole + rational model, whichever
calibrate_insta360 selected.
"""

import cv2
import numpy as np


class Insta360Undistorter:

    def __init__(self, K, D, image_size, model='fisheye', balance=0.0, lens=None):
        """
        Args:
            K: 3x3 camera matrix.
            D: Distortion coefficients (4 for fisheye, 8+ for rational).
            image_size: (width, height) of the lens image the calibration was made on.
            model: 'fisheye' or 'rational'.
            balance: 0.0 crops to valid pixels only (no black border), 1.0 keeps the
                whole field of view (black corners). Values in between trade off the two.
            lens: 'front' or 'back', which half of the stacked frame was calibrated.
        """
        if model not in ('fisheye', 'rational'):
            raise ValueError(f"model must be 'fisheye' or 'rational', got {model!r}")
        self.K = np.asarray(K, np.float64)
        self.D = np.asarray(D, np.float64)
        self.image_size = tuple(int(v) for v in image_size)
        self.model = model
        self.balance = balance
        self.lens = lens

        if model == 'fisheye':
            self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, self.image_size, np.eye(3), balance=balance)
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.new_K, self.image_size, cv2.CV_16SC2)
        else:
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(
                self.K, self.D, self.image_size, balance, self.image_size)
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.new_K, self.image_size, cv2.CV_16SC2)

    @classmethod
    def from_file(cls, path, balance=0.0):
        """Load a calibration written by calibrate_insta360."""
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(path)
        K = fs.getNode('K').mat()
        D = fs.getNode('D').mat()
        size = (int(fs.getNode('image_width').real()), int(fs.getNode('image_height').real()))
        model = fs.getNode('model').string()
        lens = fs.getNode('lens').string() or None
        fs.release()
        return cls(K, D, size, model=model, balance=balance, lens=lens)

    def undistort_frame(self, frame, interpolation=cv2.INTER_LINEAR):
        """Undistort a lens image (H, W[, C]). Must match the calibrated image size."""
        if (frame.shape[1], frame.shape[0]) != self.image_size:
            raise ValueError(f"Frame is {frame.shape[1]}x{frame.shape[0]}, "
                             f"calibration is {self.image_size[0]}x{self.image_size[1]}")
        return cv2.remap(frame, self.map1, self.map2, interpolation, borderMode=cv2.BORDER_CONSTANT)

    def undistort_mask(self, mask):
        """Undistort a label mask with nearest-neighbour so it stays binary."""
        return self.undistort_frame(mask, interpolation=cv2.INTER_NEAREST)

    def get_metadata(self):
        return {
            'model': self.model,
            'lens': self.lens,
            'K': self.K.tolist(),
            'D': self.D.ravel().tolist(),
            'new_K': self.new_K.tolist(),
            'image_size': self.image_size,
            'balance': self.balance,
        }
