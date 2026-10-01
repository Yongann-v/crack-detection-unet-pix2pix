# Claude Code Quick Start: Insta360 Undistortion Pipeline

**Read first:** `claude.md` (comprehensive implementation guide)  
**Reference:** `360_Camera_Undistortion_Methodology.pdf` (mathematical background)

---

## 🚀 5-Minute Overview

### Problem
Raw Insta360 ONE R 360° images have severe barrel distortion → YOLOv8 detection fails (78% mAP). 
Undistortion fixes this: **78% → 91% mAP (+13%)**, **±45px error → ±12px** (3.75× better).

### Solution
Pre-compute a pixel remap once, then apply fast bilinear interpolation per frame (12ms on Jetson).

### Your 5 Tasks (In Order)

| # | Task | File | Time | Complexity |
|---|------|------|------|-----------|
| 1 | Verify/create calibration (K, D) | `calibration_data/` | 30min | ⭐ |
| 2 | Implement `Insta360Undistorter` class | `undistort.py` | 1.5h | ⭐⭐ |
| 3 | Create ROS 2 subscriber node | `undistort_node.py` | 1h | ⭐⭐ |
| 4 | Integrate into YOLOv8 pipeline | `inference.py` | 1h | ⭐ |
| 5 | Test + benchmark on Jetson | `test_undistortion.py` | 1.5h | ⭐⭐ |
| **Total** | — | — | **5 hours** | — |

---

## 📦 Key Files You'll Create

### 1. `src/insta360_undistortion/undistort.py` (Core)

```python
class Insta360Undistorter:
    """Main class. Three methods:"""
    
    def __init__(self, K, D, image_size=(1920,1080), pole_crop_percent=0.10):
        """Pre-compute remap (one-time, ~1ms)"""
        self.map_x, self.map_y = cv2.fisheye.initUndistortRectifyMap(...)
    
    def undistort_frame(self, frame):
        """Apply remap + crop poles (12ms per frame)"""
        undistorted = cv2.remap(frame, self.map_x, self.map_y, ...)
        return undistorted[crop_y_start:crop_y_end, :]
    
    def get_metadata(self):
        """Return K, D, image_size as dict for logging"""
```

**Key Point:** Use `cv2.fisheye` (Kannala-Breitenstein model), NOT `cv2.calibrateCamera` (perspective).

---

### 2. `src/insta360_undistortion/undistort_node.py` (ROS 2)

```python
class UndistortionNode(Node):
    """ROS 2 node. Two methods:"""
    
    def __init__(self):
        """Load K, D; create subscriber(/camera/image_raw) + publisher(/camera/undistorted)"""
    
    def image_callback(self, msg):
        """Undistort each frame and publish"""
```

**Key Point:** Subscriber filters raw frames → undistortion → publisher sends undistorted stream.

---

### 3. `calibration_data/` (NumPy Files)

Expected after calibration:
```
insta360_oner_K.npy    [3, 3] = [[800, 0, 960], [0, 800, 540], [0, 0, 1]]
insta360_oner_D.npy    [4,]   = [-0.15, 0.05, -0.02, 0.01]
calibration_log.txt           = "RMSE: 0.68px, 35 images, 2026-10-01"
```

---

## ✅ Implementation Checklist

### Step 1: Calibration (If Needed)
- [ ] Check if `calibration_data/insta360_oner_K.npy` exists
- [ ] If not: gather 30+ Insta360 checkerboard images
- [ ] Run `cv2.fisheye.calibrate()` (Reference: `claude.md` TASK 1)
- [ ] Verify RMSE < 0.7px
- [ ] Save K.npy and D.npy

### Step 2: Undistortion Class
- [ ] Create `src/insta360_undistortion/undistort.py`
- [ ] Implement `Insta360Undistorter.__init__()` with remap
- [ ] Implement `undistort_frame()` method
- [ ] Add `get_metadata()` for logging
- [ ] Test on 10 sample frames

### Step 3: ROS 2 Node
- [ ] Create `src/insta360_undistortion/undistort_node.py`
- [ ] Implement subscriber to `/camera/image_raw`
- [ ] Implement publisher to `/camera/undistorted`
- [ ] Test with `ros2 run ... undistort_node`

### Step 4: Integration
- [ ] Modify your YOLOv8 inference script
- [ ] Load `Insta360Undistorter` at startup
- [ ] Apply `undistort_frame()` before YOLOv8.predict()
- [ ] Measure latency (target: 12ms undistortion + 18ms YOLOv8 = 31ms total)

### Step 5: Testing
- [ ] Unit test: remap correctness
- [ ] Integration test: undistort checkerboard → verify lines are straight
- [ ] Performance test: latency <12ms on Jetson Xavier NX
- [ ] Accuracy test: mAP raw (78%) vs undistorted (91%)

### Step 6: Documentation
- [ ] Write `docs/CALIBRATION_GUIDE.md`
- [ ] Update README with undistortion pipeline diagram
- [ ] Commit to `insta360-gpu-viz` branch

---

## 🔧 Code Snippets (Copy-Paste Ready)

### Load Calibration

```python
import numpy as np
from src.insta360_undistortion.undistort import Insta360Undistorter

K = np.load('calibration_data/insta360_oner_K.npy')
D = np.load('calibration_data/insta360_oner_D.npy')
undistorter = Insta360Undistorter(K, D)
```

### Undistort a Frame

```python
import cv2

raw_frame = cv2.imread('raw_insta360.jpg')
undistorted = undistorter.undistort_frame(raw_frame)
cv2.imwrite('undistorted.jpg', undistorted)
```

### Integrate with YOLOv8

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# BEFORE: raw_results = model.predict(raw_frame)
# AFTER:
undistorted = undistorter.undistort_frame(raw_frame)
results = model.predict(undistorted, conf=0.6)
```

---

## 📊 Expected Results

After implementation:

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| **mAP** (detection accuracy) | 78% | 91% | ✓ |
| **Bbox error** (localization) | ±45px | ±12px | ✓ |
| **Pole detection** (recall) | 40% | 92% | ✓ |
| **Latency** (per frame) | — | 31ms | <32ms ✓ |
| **FPS** (Jetson Xavier NX) | — | 25-32 Hz | ≥25 Hz ✓ |

---

## 🚨 Common Mistakes (Avoid These)

❌ **Don't use `cv2.calibrateCamera()`** — it's for perspective lenses.  
→ Use `cv2.fisheye.calibrate()` instead.

❌ **Don't apply distortion model inversion per pixel.**  
→ Pre-compute remap in `__init__`, reuse every frame.

❌ **Don't forget pole cropping.**  
→ Top/bottom 10% of ERP images have 10:1 area distortion.

❌ **Don't test on laptop CPU.**  
→ Jetson Xavier NX is 5× slower. GPU acceleration is essential.

❌ **Don't train YOLOv8 on distorted but run on undistorted (or vice versa).**  
→ Domain mismatch → poor mAP. Be consistent.

---

## 📚 Reference Documents

1. **`claude.md`** — Full implementation guide (8 sections, 250 lines)
2. **`360_Camera_Undistortion_Methodology.pdf`** — Mathematical background (13 pages)
   - § 3.3: Fisheye distortion equation
   - § 5.2: Undistortion algorithm
   - § 7: Error metrics & expected improvements

---

## 🎯 Success Criteria

You're done when:

✅ `undistort.py` exists with `Insta360Undistorter` class  
✅ `undistort_node.py` publishes to `/camera/undistorted`  
✅ YOLOv8 inference runs on undistorted frames  
✅ Latency measured: 12ms undistortion on Jetson Xavier NX  
✅ mAP benchmark: 91% (vs 78% baseline)  
✅ All tests pass (unit + integration + performance)  
✅ Code committed to `insta360-gpu-viz` branch  

---

## 💡 Pro Tips

1. **Pre-compute remap in `__init__`** — It's static, so compute once and store.
2. **Use OpenCV GPU acceleration** — `cv2.remap()` runs on CUDA automatically.
3. **Log latency per frame** — Add `time.perf_counter()` to find bottlenecks.
4. **Validate on real Jetson hardware** — Laptop benchmarks are misleading.
5. **Save calibration as NumPy arrays** — Faster I/O than JSON.

---

## 🆘 If You Get Stuck

| Issue | Solution |
|-------|----------|
| RMSE > 1.0px after calibration | More calibration images needed (30→60); better angles |
| Undistorted image has weird edge artifacts | Use `borderMode=cv2.BORDER_REFLECT` in remap |
| Latency > 12ms on Jetson | Check GPU utilization; may need CUDA optimization |
| mAP doesn't improve | Verify K, D are loaded correctly; check pole cropping |
| ROS 2 node won't start | Check `import cv_bridge` installed; verify image topic name |

---

**Status:** Ready to start ✓  
**Estimated Time:** 5 hours  
**Priority:** 🔴 HIGH (critical for <50mm defect localization)

Good luck! 🚀
