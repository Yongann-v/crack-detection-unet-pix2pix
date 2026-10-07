# 🚀 START HERE: Insta360 Undistortion Task Package

**Updated:** 2026-10-07  
**Project:** Concrete Pipe Defect Detection (360° Camera + LiDAR SLAM)  
**Detector:** UNet + Pix2Pix crack segmentation (`crack_detection` package)  
**Branch:** `insta360-gpu-viz`

---

## 📍 Status

| Task | State |
|------|-------|
| 1. Calibrate front and back lenses | ✅ Done (0.32 px / 0.195 px RMS) |
| 2. `Insta360Undistorter` module | ✅ Done (0.45 ms per frame on the dev PC) |
| 3+4. Undistort inside `crack_detection_node` | ✅ Done (29.4 Hz front / 30.3 Hz back, live) |
| 5. Accuracy evaluation, raw vs undistorted | ⏳ Needs labeled pipe images |
| 6. Both lenses at once | ⏳ Optional |
| 7. Jetson Xavier NX benchmark | ⏳ Not run |

---

## 📦 Files (read in this order)

1. **`claude.md`**: full record of what was built, measured results, remaining tasks
2. **`CLAUDE_CODE_QUICK_START.md`**: commands to run, recalibrate, troubleshoot
3. **`README_CLAUDE_CODE_PACKAGE.md`**: where every file lives
4. **`test_undistortion_template.py`**: pytest suite for the undistorter (runs against the real calibrations)
5. **`undistortion_pipeline.launch.py`**: launches camera publisher + crack detection with undistortion
6. **`360_Camera_Undistortion_Methodology.pdf`**: background math (describes the original
   single-fisheye/YOLOv8 plan; its numbers are not measured results)

---

## 🧠 30-Second Summary

- The Insta360 ONE R webcam stream is **two 1920×540 lens strips stacked** (back on top, front on
  bottom), already partly flattened by the firmware. Residual barrel distortion remains.
- Each lens has its own fisheye calibration in `crack_detection/calibration_data/insta360_oner_<lens>.yaml`.
- `crack_detection_node` undistorts each frame right after decoding, before zoom and UNet (+ Pix2Pix)
  inference, when `undistort_enabled` is true (the default). Balance 0.5.
- Results are crack masks, so accuracy is measured with mask IoU / F1 and crack-center error, not mAP.

---

## ✅ Definition of Done (remaining)

- [ ] Raw vs undistorted IoU / F1 measured on hand-labeled Insta360 pipe images (Task 5)
- [ ] Test template moved to `crack_detection/test/` and passing in CI
- [ ] Jetson Xavier NX: ≥25 Hz end-to-end in the chosen mode (Task 7)
- [ ] Changes committed to `insta360-gpu-viz`

---

## 🚨 Critical Reminders

❌ **DON'T:**
- Calibrate on the stacked 1920×1080 frame → one lens strip at a time
- Apply a pole crop → the strips are not equirectangular
- Run RealSense with undistortion on → use `undistort_enabled:=false`
- Mix distorted and undistorted data between training and inference
- Undistort label masks with bilinear interpolation → use `undistort_mask()` (nearest)

✅ **DO:**
- Rebuild the package after recalibrating so the YAML is installed
- Use `PYTHONNOUSERSITE=1` for the capture GUI (pip OpenCV is headless)
- Benchmark on the Jetson before trusting latency numbers for the robot
