# 📦 Claude Code Undistortion Task Package

**For:** Insta360 ONE R lens undistortion before UNet + Pix2Pix crack segmentation  
**Project:** Concrete Pipe Defect Detection (360° Camera + LiDAR SLAM)  
**Repository:** https://github.com/Yongann-v/crack-detection-unet-pix2pix/tree/insta360-gpu-viz  
**Owner:** Yong Ann VOEURN, Georgia Southern University  
**Updated:** 2026-10-07

---

## 📋 Files in This Folder

| File | Purpose |
|------|---------|
| `00_START_HERE.md` | Status board and reading order |
| `claude.md` | Full record: camera facts, what was built, measured results, remaining tasks |
| `CLAUDE_CODE_QUICK_START.md` | Commands: run, recalibrate, use in Python, troubleshoot |
| `README_CLAUDE_CODE_PACKAGE.md` | This file: where everything lives |
| `test_undistortion_template.py` | Pytest suite for `Insta360Undistorter` against the real calibrations |
| `undistortion_pipeline.launch.py` | Starts `insta360_publisher` + `crack_detection_node` for one lens |
| `360_Camera_Undistortion_Methodology.pdf` | Background math (original single-fisheye/YOLOv8 plan; not measured results) |

---

## 🎯 What Was Built

**Problem:** Insta360 lens strips have barrel distortion, so cracks near the edges are bent and
stretched before UNet sees them, and crack centers are reported in distorted coordinates.

**Solution:** Per-lens fisheye calibration, then one pre-computed `cv2.remap` per frame inside
`crack_detection_node`, before zoom and UNet (+ Pix2Pix) inference.

**Measured:**
- Calibration RMS: front 0.32 px, back 0.195 px
- Undistortion: 0.45 ms per 1920×540 frame (dev PC CPU)
- Node: 29.4 Hz (front) / 30.3 Hz (back) with undistortion, UNet fast mode (RTX 2000 Ada)
- Accuracy (IoU / F1, raw vs undistorted): **not yet measured** (Task 5)

---

## 📁 Code Locations

```
crack-detection-unet-pix2pix/
├── Agent_md/                              # this package (docs + templates)
├── crack_detection/
│   ├── crack_detection/
│   │   ├── insta360_lens.py               # stacked-frame layout, crop_lens()
│   │   ├── capture_calibration.py         # ros2 run crack_detection capture_calibration
│   │   ├── calibrate_insta360.py          # ros2 run crack_detection calibrate_insta360
│   │   ├── undistort.py                   # Insta360Undistorter
│   │   └── crack_detection_node.py        # undistort_enabled / calibration_file / undistort_balance
│   ├── calibration_data/
│   │   ├── insta360_oner_front.yaml (+ _preview.jpg)
│   │   └── insta360_oner_back.yaml  (+ _preview.jpg)
│   ├── config/crack_detection_params.yaml # undistortion on by default, front lens, balance 0.5
│   ├── launch/crack_detection.launch.py   # matching launch arguments
│   └── setup.py                           # entry points + installs calibration_data/*.yaml
└── insta360_ros/                          # camera publisher (front/back/compressed topics)

~/crack_ws/calib_images/{front,back}/      # checkerboard captures (not in the repo)
~/crack_ws/checkerboard_9x6_20mm.png       # printable board
```

---

## 🔧 Design Decisions

| Decision | Reason |
|----------|--------|
| Calibrate each 1920×540 strip separately | The stream is two stacked, firmware-flattened lenses, not one fisheye circle |
| Fisheye model (auto-selected) | 0.32 / 0.195 px vs 0.80 / 0.91 px for pinhole + rational |
| No pole crop | Only applies to equirectangular images |
| Undistort inside the detection node, no separate node | Avoids an extra image hop (~3 MB per frame); one place for coordinates |
| Balance 0.5 | Keeps most of the field of view; black arcs are never predicted as cracks |
| One YAML per lens (`cv2.FileStorage`) | Holds K, D, model, lens and size together; replaces separate `K.npy` / `D.npy` |
| Stacked frames cropped to the calibrated lens | Lets the compressed full-frame topic work with undistortion |

---

## 🔜 Remaining Work

1. **Task 5:** label crack masks on raw Insta360 pipe frames; compare IoU / F1 / crack-center error
   raw vs undistorted for UNet fast, tiled, and UNet + Pix2Pix.
2. Move `test_undistortion_template.py` to `crack_detection/test/test_undistortion.py`.
3. **Task 6 (optional):** two node instances with namespaced outputs to run both lenses.
4. **Task 7:** Jetson Xavier NX benchmark, ≥25 Hz target.
5. Update `docs/360_Camera_Undistortion_Methodology.tex` to the measured setup if it will be published.
