#!/usr/bin/env python3
"""2x2 viewer for scripts/run_dual_lens.sh: front/back lens x original/undistorted.

Shows only the overlay panel of each crack_detection_node visualization, with an info bar per
view (status, FPS, crack %, temporal count, detection events, frame) and a settings footer.
Keys: q / Esc = quit.
"""
import sys
# pip's opencv-python-headless cannot open windows; prefer Ubuntu's python3-opencv
sys.path.insert(0, '/usr/lib/python3/dist-packages')
import argparse
import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

TAGS = ['front_raw', 'front_undist', 'back_raw', 'back_undist']
LABELS = {'front_raw': 'FRONT lens - original', 'front_undist': 'FRONT lens - undistorted',
          'back_raw': 'BACK lens - original', 'back_undist': 'BACK lens - undistorted'}
TW, TH, BAR = 960, 270, 54   # tile image size and per-tile info bar height
INFO_H = 120                 # node's own info panel, cropped off
WIN = 'Insta360 crack detection'
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Rate:
    """Exponentially smoothed message rate."""
    def __init__(self):
        self.t, self.fps = None, 0.0

    def tick(self):
        now = time.monotonic()
        if self.t is not None and now > self.t:
            self.fps = 0.9 * self.fps + 0.1 / (now - self.t) if self.fps else 1.0 / (now - self.t)
        self.t = now

    def value(self):
        return 0.0 if self.t is None or time.monotonic() - self.t > 2.0 else self.fps


class Viewer(Node):
    def __init__(self):
        super().__init__('quad_view')
        self.img, self.res, self.rate = {}, {}, {t: Rate() for t in TAGS}
        self.cam_rate, self.hits = Rate(), {t: 0 for t in TAGS}
        for t in TAGS:
            self.create_subscription(CompressedImage, f'/{t}/visualization/compressed',
                                     lambda m, t=t: self.on_img(t, m), qos_profile_sensor_data)
            self.create_subscription(String, f'/{t}/result', lambda m, t=t: self.on_res(t, m), 10)
        self.create_subscription(CompressedImage, '/insta360/image_raw/compressed',
                                 lambda m: self.cam_rate.tick(), qos_profile_sensor_data)

    def on_img(self, t, m):
        self.img[t] = m
        self.rate[t].tick()

    def on_res(self, t, m):
        r = dict(kv.split('=', 1) for kv in m.data.split(',') if '=' in kv)
        if r.get('detected') == 'True' and self.res.get(t, {}).get('detected') != 'True':
            self.hits[t] += 1  # count rising edges = separate detection events
        self.res[t] = r


def text(im, s, org, scale, color, thick=2):
    cv2.putText(im, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(im, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def tile(v, t):
    m = v.img.get(t)
    if m is None:
        im = np.zeros((TH, TW, 3), np.uint8)
        text(im, 'waiting for frames...', (20, TH // 2), 0.8, (200, 200, 200))
        frame_w = frame_h = None
    else:
        im = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
        # Node layout: info panel over [original | heatmap | overlay]; keep the overlay only
        im = im[INFO_H:, 2 * im.shape[1] // 3:]
        frame_h, frame_w = im.shape[:2]
        im = cv2.resize(im, (TW, TH))

    r = v.res.get(t, {})
    detected = r.get('detected') == 'True'
    pct = float(r.get('crack_percent', 0))
    if detected and frame_w:
        cx = int(int(r.get('center_x', 0)) * TW / frame_w)
        cy = int(int(r.get('center_y', 0)) * TH / frame_h)
        cv2.drawMarker(im, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.circle(im, (cx, cy), 22, (0, 255, 255), 2)
    cv2.rectangle(im, (0, 0), (TW - 1, TH - 1), (0, 0, 255) if detected else (60, 60, 60), 4 if detected else 1)

    bar = np.full((BAR, TW, 3), 25, np.uint8)
    text(bar, LABELS[t], (10, 22), 0.6, (0, 255, 255))
    status, color = ('CRACK DETECTED', (0, 0, 255)) if detected else ('NO CRACK', (0, 220, 0))
    (sw, _), _ = cv2.getTextSize(status, FONT, 0.75, 2)
    text(bar, status, (TW - sw - 12, 24), 0.75, color)
    text(bar, f"FPS {v.rate[t].value():4.1f}   Crack {pct:5.2f}%   "
              f"Temporal {r.get('temporal_status', '-').split(' ')[0]}   "
              f"Events {v.hits[t]}   Frame {r.get('frame', '-')}",
         (10, 45), 0.5, (230, 230, 230), 1)
    return np.vstack([bar, im])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--threshold', type=float, default=0.2, help='Shown in the footer only')
    ap.add_argument('--min-crack-percent', type=float, default=2.0, help='Shown in the footer only')
    ap.add_argument('--model', default='UNet + Pix2Pix', help='Shown in the footer only')
    ap.add_argument('--snapshot', help='Save one rendered frame here ~6 s after start')
    args = ap.parse_args()

    rclpy.init()
    v = Viewer()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1600, 570)
    snap = args.snapshot
    start = time.monotonic()
    # Spin in the background so message rates are measured on arrival, not per redraw
    threading.Thread(target=rclpy.spin, args=(v,), daemon=True).start()
    while rclpy.ok():
        t = [tile(v, x) for x in TAGS]
        grid = np.vstack([np.hstack(t[:2]), np.hstack(t[2:])])
        foot = np.full((34, grid.shape[1], 3), 15, np.uint8)
        text(foot, f"Model {args.model}   |   Threshold {args.threshold:.0%}   |   Min crack {args.min_crack_percent}%   |   "
                   f"Camera {v.cam_rate.value():4.1f} FPS   |   Left: original   Right: undistorted"
                   f"   |   q = quit", (10, 23), 0.55, (230, 230, 230), 1)
        out = np.vstack([grid, foot])
        if snap and time.monotonic() - start > 6:
            cv2.imwrite(snap, out)
            snap = None
        cv2.imshow(WIN, out)
        if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
            break


if __name__ == '__main__':
    main()
