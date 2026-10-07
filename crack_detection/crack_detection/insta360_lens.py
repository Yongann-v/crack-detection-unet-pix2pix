"""Insta360 ONE R webcam-mode frame layout shared by capture, calibration and undistortion.

In USB webcam mode the camera streams a 1920x1080 MJPG frame with both lenses stacked:
the back lens in the top half and the front lens in the bottom half (1920x540 each).
"""

LENSES = ('front', 'back')


def crop_lens(frame, lens):
    """Return the half of a stacked dual-lens frame that belongs to `lens`."""
    h = frame.shape[0] // 2
    if lens == 'back':
        return frame[:h]
    if lens == 'front':
        return frame[h:]
    raise ValueError(f"lens must be one of {LENSES}, got {lens!r}")
