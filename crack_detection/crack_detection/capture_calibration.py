#!/usr/bin/env python3
"""Capture checkerboard images from one Insta360 ONE R lens for fisheye calibration.

Opens the camera directly (stop insta360_publisher first, it holds /dev/video0).

Usage:
    ros2 run crack_detection capture_calibration --lens front --out calib_images/front
    ros2 run crack_detection capture_calibration --make-board checkerboard.png

Keys in the preview window:
    space  save the current frame (only when the board is detected)
    a      toggle auto-save (saves when the board has moved to a new spot)
    q/Esc  quit

Aim for 30+ images. Cover the whole lens, especially the left/right edges where
distortion is strongest, and tilt the board at different angles and distances.
"""

import argparse
import os
import time

import cv2
import numpy as np

from crack_detection.insta360_lens import LENSES, crop_lens

DETECT_SCALE = 0.5          # Detect on a half-size frame to keep the preview live
AUTO_MIN_MOVE_PX = 120      # Auto-save only when the board centre moved this far from every saved pose
AUTO_MIN_INTERVAL_S = 1.0


def parse_board(text):
    cols, rows = (int(v) for v in text.lower().split('x'))
    return cols, rows


def make_board(path, board, square_mm, dpi=300):
    """Write a printable checkerboard with `board` inner corners and a white margin."""
    cols, rows = board
    px = int(round(square_mm / 25.4 * dpi))
    squares = np.indices((rows + 1, cols + 1)).sum(axis=0) % 2
    img = np.kron(squares, np.ones((px, px))).astype(np.uint8) * 255
    img = cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, img)
    w_mm = (cols + 3) * square_mm
    h_mm = (rows + 3) * square_mm
    print(f"Wrote {path}: {cols}x{rows} inner corners, {square_mm} mm squares, "
          f"{w_mm:.0f}x{h_mm:.0f} mm page at {dpi} dpi.")
    print("Print at 100% scale (no 'fit to page'), then measure a square and pass "
          "the measured size to calibrate_insta360 --square-mm.")


def open_camera(device, width, height):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {device}. Is insta360_publisher still running?")
    return cap


def detect(gray, board):
    small = cv2.resize(gray, None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(small, board, flags)
    return (corners / DETECT_SCALE) if found else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='/dev/video0')
    ap.add_argument('--lens', choices=LENSES, default='front')
    ap.add_argument('--board', default='9x6', help='Inner corners, cols x rows (default 9x6)')
    ap.add_argument('--out', default=None, help='Output directory (default calib_images/<lens>)')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--make-board', metavar='PNG', help='Only write a printable checkerboard and exit')
    ap.add_argument('--square-mm', type=float, default=25.0, help='Square size for --make-board')
    args = ap.parse_args()

    board = parse_board(args.board)
    if args.make_board:
        make_board(args.make_board, board, args.square_mm)
        return

    out_dir = args.out or os.path.join('calib_images', args.lens)
    os.makedirs(out_dir, exist_ok=True)
    saved = len([f for f in os.listdir(out_dir) if f.endswith('.png')])

    cap = open_camera(args.device, args.width, args.height)
    coverage = None
    saved_centres = []
    auto = False
    last_save = 0.0
    win = f'Insta360 calibration capture ({args.lens} lens)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"Saving to {out_dir} ({saved} images already there). space=save, a=auto, q=quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, retrying...")
            time.sleep(0.1)
            continue
        lens_img = crop_lens(frame, args.lens)
        if coverage is None:
            coverage = np.zeros(lens_img.shape[:2], np.uint8)
        gray = cv2.cvtColor(lens_img, cv2.COLOR_BGR2GRAY)
        corners = detect(gray, board)

        view = lens_img.copy()
        # Tint areas already covered by saved boards so gaps are easy to see
        view[coverage > 0] = (0.6 * view[coverage > 0] + [0, 90, 0]).astype(np.uint8)
        if corners is not None:
            cv2.drawChessboardCorners(view, board, corners.astype(np.float32), True)

        want_save = False
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('a'):
            auto = not auto
        if key == ord(' ') and corners is not None:
            want_save = True
        if auto and corners is not None and time.time() - last_save > AUTO_MIN_INTERVAL_S:
            c = corners.reshape(-1, 2).mean(axis=0)
            want_save = all(np.linalg.norm(c - s) > AUTO_MIN_MOVE_PX for s in saved_centres)

        if want_save:
            path = os.path.join(out_dir, f'{args.lens}_{saved:03d}.png')
            cv2.imwrite(path, lens_img)
            saved += 1
            last_save = time.time()
            saved_centres.append(corners.reshape(-1, 2).mean(axis=0))
            cv2.fillConvexPoly(coverage, cv2.convexHull(corners.astype(np.int32)), 255)
            print(f"Saved {path}")

        status = (f"saved: {saved}  board: {'YES' if corners is not None else 'no'}  "
                  f"auto: {'ON' if auto else 'off'}  coverage: {100 * (coverage > 0).mean():.0f}%")
        cv2.putText(view, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(view, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(win, view)

    cap.release()
    cv2.destroyAllWindows()
    print(f"{saved} images in {out_dir}")


if __name__ == '__main__':
    main()
