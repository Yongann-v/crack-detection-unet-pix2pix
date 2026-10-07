# Claude Code Quick Start: Insta360 Undistortion

**Full details:** `claude.md`

All commands assume:
```bash
cd ~/crack_ws && source /opt/ros/jazzy/setup.bash && source install/setup.bash
```

---

## ▶️ Run Crack Detection with Undistortion

Undistortion is on by default (front lens, balance 0.5).

```bash
ros2 run insta360_ros insta360_publisher                  # terminal 1
ros2 launch crack_detection crack_detection.launch.py     # terminal 2 (front lens)
```

Back lens:
```bash
ros2 launch crack_detection crack_detection.launch.py \
  camera_topic:=/insta360/back/image_raw \
  calibration_file:=calibration_data/insta360_oner_back.yaml
```

Compressed full frame (node crops it to the calibrated lens, then undistorts):
```bash
ros2 launch crack_detection crack_detection.launch.py camera_topic:=/insta360/image_raw/compressed
```

RealSense (no Insta360 calibration):
```bash
ros2 launch crack_detection crack_detection.launch.py undistort_enabled:=false \
  camera_topic:=/camera/camera/color/image_raw ...
```

Or start both camera and detection at once with the template in this folder:
```bash
ros2 launch Agent_md/undistortion_pipeline.launch.py lens:=front
```

---

## 🎚️ Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `undistort_enabled` | `true` | `false` for RealSense |
| `calibration_file` | `calibration_data/insta360_oner_front.yaml` | Relative to the package share dir |
| `undistort_balance` | `0.5` | 0 = crop to valid pixels (~40% FOV lost), 1 = full FOV, big black corners |

---

## 🔁 Recalibrate a Lens

1. Print `~/crack_ws/checkerboard_9x6_20mm.png` at 100% (or regenerate:
   `ros2 run crack_detection capture_calibration --make-board board.png --square-mm 20`).
   Measure a square.
2. Stop `insta360_publisher` (the capture tool opens `/dev/video0` itself).
3. Capture 40+ images, covering corners and edges:
   ```bash
   PYTHONNOUSERSITE=1 ros2 run crack_detection capture_calibration --lens front --board 9x6
   ```
   Keys: space = save, a = auto-save, q = quit. Green tint = covered area.
4. Calibrate:
   ```bash
   ros2 run crack_detection calibrate_insta360 --images calib_images/front --lens front \
       --board 9x6 --square-mm 20 --out src/crack-detection-unet-pix2pix/crack_detection/calibration_data
   ```
   Aim for RMS < 0.5 px. Check `insta360_oner_<lens>_preview.jpg`.
5. Rebuild: `colcon build --packages-select crack_detection --symlink-install`

---

## 🐍 Use the Undistorter in Python

```python
from crack_detection.undistort import Insta360Undistorter
from crack_detection.insta360_lens import crop_lens

u = Insta360Undistorter.from_file('calibration_data/insta360_oner_front.yaml', balance=0.5)
lens_img = crop_lens(stacked_1920x1080_frame, u.lens)    # 1920x540
undistorted = u.undistort_frame(lens_img)
mask_u = u.undistort_mask(label_mask)                     # nearest-neighbour for labels
```

---

## 🧪 Tests

```bash
cd ~/crack_ws/src/crack-detection-unet-pix2pix
python3 -m pytest Agent_md/test_undistortion_template.py -v -s
```

34 tests across both lenses: calibration quality, remap tables, straight-line checks, mask
handling, latency (<12 ms) and a UNet smoke test. No `PYTHONNOUSERSITE` here: torch is in `~/.local`.

---

## 📊 Measured Results

| Item | Front | Back |
|------|-------|------|
| Calibration RMS | 0.32 px (79 images) | 0.195 px (82 images) |
| Undistortion, 1920×540 | 0.45 ms (dev PC CPU) | same maps, same cost |
| Node rate, UNet fast + undistortion | 29.4 Hz (RTX 2000 Ada) | 30.3 Hz (via launch template) |
| Mask IoU / F1, raw vs undistorted | Task 5, not measured | Task 5 |

---

## 🆘 Troubleshooting

| Issue | Fix |
|-------|-----|
| `cv2.error ... The function is not implemented` on `namedWindow` | Prefix with `PYTHONNOUSERSITE=1` (pip OpenCV is headless) |
| `Cannot open /dev/video0` in capture tool | Stop `insta360_publisher` |
| Node logs `Frame is WxH but calibration is 1920x540` | Wrong camera for this calibration; set `undistort_enabled:=false` or pick the right lens file |
| Calibration RMS > 1 px | More images near the edges; check `--board` and `--square-mm` |
| New calibration not picked up | Rebuild so the YAML is copied into `install/` |
| Black arcs at top/bottom of output | Expected with balance 0.5; lower balance crops them (and field of view) |
