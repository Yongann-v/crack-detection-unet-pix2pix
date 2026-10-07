#!/usr/bin/env python3
"""Calibrate one Insta360 ONE R lens from checkerboard images.

Fits the OpenCV fisheye (Kannala-Brandt) model and the pinhole + rational model, and keeps the
one with the lower reprojection error unless --model is given. The camera firmware already
partly flattens each lens in webcam mode, so either model may fit better.

Usage:
    ros2 run crack_detection calibrate_insta360 --images calib_images/front --lens front \\
        --board 9x6 --square-mm 25 --out calibration_data

Writes <out>/insta360_oner_<lens>.yaml (K, D, model, image size, RMS) and
<out>/insta360_oner_<lens>_preview.jpg (raw vs undistorted side by side).
"""

import argparse
import glob
import os
import re
from datetime import datetime

import cv2
import numpy as np

from crack_detection.capture_calibration import parse_board
from crack_detection.insta360_lens import LENSES
from crack_detection.undistort import Insta360Undistorter

OUTLIER_FACTOR = 3.0      # Drop images whose error exceeds this multiple of the median, then refit
RATIONAL_MIN_GAIN = 0.9   # Prefer fisheye unless rational RMS is at least 10% lower


def find_corners(path, board):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCornersSB(
        gray, board, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, board, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return None, gray.shape[::-1]
        corners = cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
    return corners.reshape(1, -1, 2).astype(np.float64), gray.shape[::-1]


def calibrate_fisheye(obj, img, size):
    """Fisheye calibration, dropping images OpenCV reports as ill-conditioned."""
    keep = list(range(len(obj)))
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND
             | cv2.fisheye.CALIB_FIX_SKEW)
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-6)
    while len(keep) >= 5:
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                [obj[i] for i in keep], [img[i] for i in keep], size, None, None,
                flags=flags, criteria=criteria)
            return rms, K, D, rvecs, tvecs, keep
        except cv2.error as e:
            m = re.search(r'input array (\d+)', str(e))
            if not m:
                raise
            dropped = keep.pop(int(m.group(1)))
            print(f"  fisheye: dropping ill-conditioned image #{dropped}")
    raise RuntimeError("Too few usable images for fisheye calibration")


def calibrate_rational(obj, img, size):
    flags = cv2.CALIB_RATIONAL_MODEL
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        [o.reshape(-1, 3).astype(np.float32) for o in obj],
        [i.reshape(-1, 2).astype(np.float32) for i in img], size, None, None, flags=flags)
    return rms, K, D, rvecs, tvecs, list(range(len(obj)))


def per_image_errors(model, obj, img, K, D, rvecs, tvecs, keep):
    errs = []
    for j, i in enumerate(keep):
        if model == 'fisheye':
            proj, _ = cv2.fisheye.projectPoints(obj[i], rvecs[j], tvecs[j], K, D)
        else:
            proj, _ = cv2.projectPoints(obj[i].reshape(-1, 3), rvecs[j], tvecs[j], K, D)
        errs.append(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img[i].reshape(-1, 2)) ** 2, axis=1))))
    return np.array(errs)


def fit(model, obj, img, size):
    """Calibrate, drop gross outliers, refit. Returns (rms, K, D, kept indices, per-image errors)."""
    calib = calibrate_fisheye if model == 'fisheye' else calibrate_rational
    rms, K, D, rvecs, tvecs, keep = calib(obj, img, size)
    errs = per_image_errors(model, obj, img, K, D, rvecs, tvecs, keep)
    bad = errs > OUTLIER_FACTOR * np.median(errs)
    if bad.any() and len(keep) - bad.sum() >= 5:
        print(f"  {model}: dropping {bad.sum()} outlier image(s): "
              f"{[keep[j] for j in np.flatnonzero(bad)]}")
        sub = [keep[j] for j in np.flatnonzero(~bad)]
        rms, K, D, rvecs, tvecs, kept = calib([obj[i] for i in sub], [img[i] for i in sub], size)
        keep = [sub[j] for j in kept]
        errs = per_image_errors(model, obj, img, K, D, rvecs, tvecs, keep)
    return rms, K, D, keep, errs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', required=True, help='Directory of checkerboard images from one lens')
    ap.add_argument('--lens', choices=LENSES, required=True)
    ap.add_argument('--board', default='9x6', help='Inner corners, cols x rows (default 9x6)')
    ap.add_argument('--square-mm', type=float, required=True, help='Measured square size in mm')
    ap.add_argument('--model', choices=('auto', 'fisheye', 'rational'), default='auto')
    ap.add_argument('--out', default='calibration_data')
    args = ap.parse_args()

    board = parse_board(args.board)
    objp = np.zeros((1, board[0] * board[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * args.square_mm

    paths = sorted(glob.glob(os.path.join(args.images, '*.png')) + glob.glob(os.path.join(args.images, '*.jpg')))
    if not paths:
        raise SystemExit(f"No images in {args.images}")

    obj, img, used, size = [], [], [], None
    for p in paths:
        corners, s = find_corners(p, board)
        if size is None:
            size = s
        elif s != size:
            raise SystemExit(f"{p} is {s}, expected {size}; images must all come from one lens crop")
        if corners is None:
            print(f"  no board found: {os.path.basename(p)}")
            continue
        obj.append(objp)
        img.append(corners)
        used.append(p)
    print(f"Board found in {len(used)}/{len(paths)} images ({size[0]}x{size[1]})")
    if len(used) < 10:
        raise SystemExit("Need at least 10 usable images (30+ recommended)")

    models = ('fisheye', 'rational') if args.model == 'auto' else (args.model,)
    results = {}
    for m in models:
        try:
            results[m] = fit(m, obj, img, size)
            print(f"  {m}: RMS {results[m][0]:.3f} px over {len(results[m][3])} images")
        except (cv2.error, RuntimeError) as e:
            print(f"  {m}: failed ({e})")
    if not results:
        raise SystemExit("Calibration failed for every model")
    model = min(results, key=lambda m: results[m][0])
    # The rational model has twice the parameters; only take it when it is clearly better
    if model == 'rational' and 'fisheye' in results and \
            results['rational'][0] > RATIONAL_MIN_GAIN * results['fisheye'][0]:
        model = 'fisheye'
    rms, K, D, keep, errs = results[model]

    os.makedirs(args.out, exist_ok=True)
    yaml_path = os.path.join(args.out, f'insta360_oner_{args.lens}.yaml')
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_WRITE)
    fs.write('model', model)
    fs.write('lens', args.lens)
    fs.write('image_width', size[0])
    fs.write('image_height', size[1])
    fs.write('K', K)
    fs.write('D', D)
    fs.write('rms_px', float(rms))
    fs.write('num_images', len(keep))
    fs.write('board', args.board)
    fs.write('square_mm', args.square_mm)
    fs.write('date', datetime.now().isoformat(timespec='seconds'))
    for m, r in results.items():
        fs.write(f'rms_{m}_px', float(r[0]))
    fs.release()

    worst = np.argsort(errs)[::-1][:5]
    print(f"\nSelected model: {model}  RMS {rms:.3f} px  ({len(keep)} images)")
    print(f"K =\n{K}\nD = {D.ravel()}")
    print("Worst images: " + ", ".join(f"{os.path.basename(used[keep[j]])} {errs[j]:.2f}px" for j in worst))
    print(f"Wrote {yaml_path}")

    # Side-by-side preview on the image with the most spread-out board
    undistorter = Insta360Undistorter.from_file(yaml_path)
    sample = cv2.imread(used[keep[int(np.argmin(errs))]])
    preview = np.hstack([sample, undistorter.undistort_frame(sample)])
    preview_path = os.path.join(args.out, f'insta360_oner_{args.lens}_preview.jpg')
    cv2.imwrite(preview_path, preview)
    print(f"Wrote {preview_path}")
    if rms > 1.0:
        print("RMS above 1 px: capture more images, especially near the lens edges, and recheck --square-mm/--board.")


if __name__ == '__main__':
    main()
