# Claude.md: Insta360 Lens Undistortion for UNet + Pix2Pix Crack Detection

**Project:** Concrete Pipe Defect Detection (360° Camera + LiDAR SLAM)  
**Branch:** `insta360-gpu-viz`  
**Workspace:** https://github.com/Yongann-v/crack-detection-unet-pix2pix/tree/insta360-gpu-viz  
**Owner:** Yong Ann VOEURN, Georgia Southern University  
**Last Updated:** 2026-10-07

---

## 🎯 Objective

Undistort Insta360 ONE R frames in real time before UNet + Pix2Pix crack segmentation, so cracks
keep their true shape and crack centers are reported in undistorted image coordinates.

**Status:** Calibration (both lenses) and node integration are **done**. Remaining: accuracy
evaluation on labeled pipe images (Task 5), Jetson benchmark, and optionally running both lenses
at once (Task 6).

---

## 📷 Camera Facts (measured, not assumed)

In USB webcam mode the Insta360 ONE R streams one **1920×1080 MJPG** frame with both lenses stacked:

| Half | Lens | Size | ROS topic (`insta360_publisher`) |
|------|------|------|----------------------------------|
| Top | back | 1920×540 | `/insta360/back/image_raw` |
| Bottom | front | 1920×540 | `/insta360/front/image_raw` |

The camera firmware already flattens each lens into a wide strip, so the residual distortion is
moderate barrel distortion, not a raw fisheye circle and not an equirectangular panorama. Therefore:

- Each lens is calibrated **separately on its 1920×540 strip**.
- There is **no pole crop** (that applies to equirectangular images only).
- Pixels are not square: fx ≈ 0.9 × fy (the firmware squeezes horizontally). The calibration
  models this; do not force fx = fy.

---

## 📁 What Exists

```
crack-detection-unet-pix2pix/crack_detection/
├── crack_detection/
│   ├── insta360_lens.py          # crop_lens(frame, 'front'|'back'), LENSES
│   ├── capture_calibration.py    # live checkerboard capture + printable board (ros2 run)
│   ├── calibrate_insta360.py     # fisheye vs rational fit, outlier rejection (ros2 run)
│   ├── undistort.py              # Insta360Undistorter (remap tables built once)
│   └── crack_detection_node.py   # undistorts in image_callback() when undistort_enabled
├── calibration_data/
│   ├── insta360_oner_front.yaml  # K, D, model, lens, size, RMS, board, date
│   ├── insta360_oner_front_preview.jpg
│   ├── insta360_oner_back.yaml
│   └── insta360_oner_back_preview.jpg
├── config/crack_detection_params.yaml   # undistort_enabled / calibration_file / undistort_balance
└── launch/crack_detection.launch.py     # same three launch arguments
```

Calibration images live outside the repo in `~/crack_ws/calib_images/{front,back}/`.

---

## ✅ TASK 1: Calibration (DONE)

Board: 9×6 inner corners, 20 mm squares (`~/crack_ws/checkerboard_9x6_20mm.png`, printed at 100%).

| Lens | Model | RMS | Images | fx, fy | cx, cy | D (k1..k4) |
|------|-------|-----|--------|--------|--------|------------|
| front | fisheye | 0.32 px | 79/79 | 641.8, 707.8 | 921.7, 329.1 | 0.187, −0.134, 0.097, −0.023 |
| back | fisheye | 0.195 px | 82/82 | 635.5, 696.4 | 928.0, 301.7 | 0.172, −0.121, 0.092, −0.022 |

`calibrate_insta360` fits both the OpenCV fisheye (Kannala-Brandt) model and the pinhole +
rational model and keeps fisheye unless rational is at least 10% better. Rational scored
0.80 px (front) and 0.91 px (back), so fisheye won clearly on both lenses.

Front-lens coverage is thin in the top corners, bottom row and right edge. Adding 15–20 images
there would firm up the outer edges; the back lens is evenly covered.

**To recalibrate a lens:**
```bash
# OpenCV from pip (~/.local) is headless; PYTHONNOUSERSITE=1 uses Ubuntu's python3-opencv with GUI
PYTHONNOUSERSITE=1 ros2 run crack_detection capture_calibration --lens front --board 9x6
ros2 run crack_detection calibrate_insta360 --images calib_images/front --lens front \
    --board 9x6 --square-mm 20 --out src/crack-detection-unet-pix2pix/crack_detection/calibration_data
colcon build --packages-select crack_detection --symlink-install   # installs the new YAML
```
Stop `insta360_publisher` first: the capture tool opens `/dev/video0` directly.

---

## ✅ TASK 2: Undistortion Module (DONE)

`crack_detection/undistort.py`:

```python
from crack_detection.undistort import Insta360Undistorter

u = Insta360Undistorter.from_file('calibration_data/insta360_oner_front.yaml', balance=0.5)
undistorted = u.undistort_frame(lens_img)   # lens_img must be 1920x540 (one lens)
mask_u = u.undistort_mask(label_mask)        # nearest-neighbour, stays binary
u.lens, u.model, u.image_size, u.new_K       # metadata; also u.get_metadata()
```

- Remap tables (`CV_16SC2`) are built once in `__init__`; each frame is one `cv2.remap`.
- `balance` sets the crop: 0 = zoomed to valid pixels (loses ~40% field of view), 1 = full field
  of view with large black corners. **0.5 is the chosen default** (small black arcs top and bottom;
  black pixels are never predicted as cracks).
- Measured: **0.45 ms per 1920×540 frame** on the development PC CPU (synthetic test).

---

## ✅ TASK 3 + 4: Node Integration (DONE)

There is **no separate undistortion node**. `crack_detection_node` undistorts in `image_callback()`
right after decoding, before `apply_zoom()` and `predict_frame()`. Mask, crack percentage, crack
center and saved captures are therefore all in undistorted coordinates.

| Parameter | Default (YAML / launch) | Meaning |
|-----------|-------------------------|---------|
| `undistort_enabled` | `true` | Turn undistortion on/off |
| `calibration_file` | `calibration_data/insta360_oner_front.yaml` | Relative paths resolve against the package share dir |
| `undistort_balance` | `0.5` | 0 = crop to valid pixels, 1 = keep full field of view |

Frame handling when enabled:
- 1920×540 lens strip → undistorted.
- 1920×1080 stacked frame (e.g. `/insta360/image_raw/compressed`) → cropped to the calibrated
  lens first, then undistorted. Detection then sees only that lens.
- Any other size (e.g. RealSense) → error logged every 5 s and the frame is skipped.
  **Use `undistort_enabled:=false` for RealSense.**

**Measured live** (front lens, UNet fast mode, fp16, RTX 2000 Ada): **29.4 Hz** on
`/crack_detection/result`, the same as without undistortion. Back lens (via
`undistortion_pipeline.launch.py lens:=back`): **30.3 Hz**.

Back lens:
```bash
ros2 launch crack_detection crack_detection.launch.py \
  camera_topic:=/insta360/back/image_raw \
  calibration_file:=calibration_data/insta360_oner_back.yaml
```

---

## 📋 TASK 5: Accuracy Evaluation (TODO)

Detection is segmentation, so use mask metrics. mAP and bounding-box error do not apply.

1. Collect Insta360 frames of real pipe/concrete cracks (both lenses if both will be used).
2. Hand-label crack masks on the **raw** lens strips.
3. Create undistorted image/mask pairs with `undistort_frame()` / `undistort_mask()` (same balance
   as the node), so both sides are scored against matching ground truth.
4. Run the same weights on raw and undistorted inputs, for UNet fast, UNet tiled and UNet + Pix2Pix.
5. Report per mode:

| Metric | Raw | Undistorted | Notes |
|--------|-----|-------------|-------|
| Crack mask IoU | measure | measure | Higher is better |
| Crack pixel F1 (Dice) | measure | measure | Higher is better |
| Edge-region crack recall | measure | measure | Left/right 20% of the strip, where distortion is worst |
| Crack-center error | measure | measure | px, vs ground-truth mask centroid |

Score undistorted predictions only inside the valid (non-black) region.

If undistorted results lag, check which images `best_model.pth` (UNet) and
`pix2pix_epoch_98_best.pth` were trained on. Fine-tune UNet on undistorted Insta360 frames, then
retrain Pix2Pix (`train_pix2pix.py`) on the new UNet outputs.

Also move `test_undistortion_template.py` (34 tests, all passing) to
`crack_detection/test/test_undistortion.py`.

---

## 📋 TASK 6: Both Lenses at Once (OPTIONAL)

One node handles one lens. Running front and back together needs two node instances with
different names and the output topics (`/crack_detection/...`) made configurable or namespaced,
otherwise both publish to the same topics.

---

## 📋 TASK 7: Jetson Xavier NX Benchmark (TODO)

All timings above are from the development PC. On the Jetson, measure per mode (UNet fast/tiled,
with/without Pix2Pix): undistortion ms, inference ms, end-to-end Hz. Target ≥25 Hz.

---

## 💬 Notes for Claude Code

1. **Calibrate per lens on the 1920×540 strip.** Never on the stacked 1920×1080 frame.
2. **Fisheye model via `cv2.fisheye`**, not `cv2.calibrateCamera`, unless the rational fit is
   clearly better (the calibration script decides).
3. **Remap once, reuse every frame.** Rebuild the undistorter only if balance or calibration changes.
4. **Keep training and inference consistent.** Same lens, same undistortion, same balance.
5. **Undistort label masks with nearest-neighbour** (`undistort_mask`), never bilinear.
6. **Rebuild after recalibrating** so the YAML is installed to the package share directory.
7. **Headless OpenCV:** `~/.local` has `opencv-python-headless`; GUI tools need `PYTHONNOUSERSITE=1`.

---

## 📝 References

1. OpenCV fisheye module: https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
2. Kannala, J., Brandt, S. S. (2006). A generic camera model and calibration method for
   conventional, wide-angle, and fish-eye lenses. IEEE TPAMI.
3. `docs/360_Camera_Undistortion_Methodology.pdf`: background math (still describes the original
   single-fisheye/YOLOv8 plan; numbers there are not measured results).
