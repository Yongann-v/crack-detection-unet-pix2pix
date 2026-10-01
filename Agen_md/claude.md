# Claude.md: Insta360 Fisheye Undistortion Pipeline

**Project:** Concrete Pipe Defect Detection (360° Camera + LiDAR SLAM)  
**Branch:** `insta360-gpu-viz`  
**Workspace:** https://github.com/Yongann-v/crack-detection-unet-pix2pix/tree/insta360-gpu-viz  
**Owner:** Yong Ann VOEURN, Georgia Southern University  
**Last Updated:** 2026-10-01

---

## 🎯 Objective

Implement **real-time fisheye undistortion pipeline** for Insta360 ONE R 360° images before YOLOv8/U-Net defect detection. Target: **<12ms latency**, **91% mAP** (vs 78% on raw distorted images), **±12px bbox error** (vs ±45px).

**Success Metric:** Integrate undistortion into ROS 2 pipeline; validate on Jetson Xavier NX at 25+ Hz.

---

## 📁 Workspace Structure

```
crack-detection-unet-pix2pix/
├── insta360_undistortion/           [NEW - your task]
│   ├── __init__.py
│   ├── calibration.py               # Camera calibration workflow
│   ├── undistort.py                 # Core undistortion module
│   ├── undistort_node.py            # ROS 2 subscriber node
│   ├── test_undistortion.py         # Unit tests
│   └── calibration_data/
│       ├── insta360_oner_K.npy      # Camera matrix K (3x3)
│       ├── insta360_oner_D.npy      # Distortion coefficients D (4,)
│       └── calibration_log.txt      # RMSE, num images, date
├── launch/
│   └── undistortion_pipeline.launch.py  # ROS 2 launch file
├── config/
│   └── undistortion_params.yaml     # Jetson Xavier tuning params
├── tests/
│   └── test_undistortion_*.py       # Pytest suite
└── docs/
    └── UNDISTORTION_TECHNICAL.md    # This implementation guide
```

---

## 📋 Core Tasks (Priority Order)

### **TASK 1: Insta360 ONE R Calibration (If Not Done)**
**Status:** Check if calibration files exist in `calibration_data/`

**If missing, implement:**
```python
# src/insta360_undistortion/calibration.py

def calibrate_insta360_checkerboard(image_dir: str, 
                                   checkerboard_size: tuple = (8, 6),
                                   square_size_mm: float = 25.0) -> tuple:
    """
    Calibrate Insta360 ONE R using checkerboard patterns.
    
    Input:  Directory with 30+ calibration images (various angles/distances)
    Output: K (3x3), D (4,), reprojection_error (float)
    
    Expected Results (Insta360 ONE R):
    • K = [[800, 0, 960], [0, 800, 540], [0, 0, 1]]  (f_x=f_y~800px)
    • D = [-0.15, 0.05, -0.02, 0.01]                (barrel + pincushion)
    • RMSE < 0.7px (good)
    """
    # Use cv2.fisheye.calibrate() (Kannala-Breitenstein model)
    # Reference: docs/360_Camera_Undistortion_Methodology.pdf
```

**Deliverable:**
- `calibration_data/insta360_oner_K.npy`
- `calibration_data/insta360_oner_D.npy`
- `calibration_data/calibration_log.txt` (RMSE, num images, date)

---

### **TASK 2: Core Undistortion Module**
**File:** `src/insta360_undistortion/undistort.py`

**Class: `Insta360Undistorter`**

```python
class Insta360Undistorter:
    """
    Real-time fisheye undistortion for Insta360 ONE R.
    
    Algorithm:
    1. Load K, D from calibration files
    2. Pre-compute remap (map_x, map_y) for all image pixels
    3. For each frame: cv2.remap(frame, map_x, map_y, INTER_LINEAR)
    4. Crop poles (skip top/bottom 10%)
    
    Latency: ~12ms per 1920x1080 frame on Jetson Xavier NX
    """
    
    def __init__(self, K: np.ndarray, D: np.ndarray, 
                 image_size: tuple = (1920, 1080),
                 pole_crop_percent: float = 0.10):
        """Initialize undistorter with calibration params."""
        self.K = K.astype(np.float32)
        self.D = D.astype(np.float32)
        self.image_size = image_size
        self.pole_crop_percent = pole_crop_percent
        
        # Pre-compute remap
        self.map_x, self.map_y = self._build_remap()
        self.crop_y_start = int(image_size[0] * pole_crop_percent)
        self.crop_y_end = int(image_size[0] * (1 - pole_crop_percent))
    
    def _build_remap(self) -> tuple:
        """
        Pre-compute pixel mapping for all undistorted (u,v) -> distorted (u_d, v_d).
        
        Uses OpenCV fisheye model:
        θ_d = r * (1 + k_1*r^2 + k_2*r^4 + k_3*r^6 + k_4*r^8)
        
        Equation reference: Technical PDF § 5.2
        """
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, 
            R=np.eye(3),
            P=self.K,
            size=self.image_size,
            m1type=cv2.CV_32F
        )
        return map_x, map_y
    
    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort raw Insta360 frame and crop poles.
        
        Input:  BGR or grayscale frame (H, W, C)
        Output: Undistorted, pole-cropped frame
        
        Timing: ~12ms on Jetson Xavier NX
        """
        # Apply remap (GPU-accelerated on CUDA)
        undistorted = cv2.remap(frame, self.map_x, self.map_y, 
                               cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_CONSTANT)
        
        # Crop poles (skip top/bottom 10% where distortion is worst)
        cropped = undistorted[self.crop_y_start:self.crop_y_end, :]
        
        return cropped
    
    def get_metadata(self) -> dict:
        """Return calibration metadata for logging."""
        return {
            'K': self.K.tolist(),
            'D': self.D.tolist(),
            'image_size': self.image_size,
            'pole_crop_percent': self.pole_crop_percent,
            'algorithm': 'cv2.fisheye (Kannala-Breitenstein)'
        }
```

**Deliverable:** Working undistortion class with unit tests

---

### **TASK 3: ROS 2 Subscriber Node**
**File:** `src/insta360_undistortion/undistort_node.py`

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from .undistort import Insta360Undistorter

class UndistortionNode(Node):
    """
    ROS 2 subscriber node for real-time undistortion.
    
    Subscribes:  /camera/image_raw (Insta360 stream)
    Publishes:   /camera/undistorted (processed image)
    """
    
    def __init__(self):
        super().__init__('insta360_undistortion_node')
        
        # Load calibration
        calib_dir = 'calibration_data'
        K = np.load(f'{calib_dir}/insta360_oner_K.npy')
        D = np.load(f'{calib_dir}/insta360_oner_D.npy')
        
        self.undistorter = Insta360Undistorter(K, D)
        self.bridge = CvBridge()
        
        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher
        self.publisher = self.create_publisher(
            Image,
            '/camera/undistorted',
            10
        )
        
        self.get_logger().info('Undistortion node initialized')
    
    def image_callback(self, msg: Image):
        """Process incoming raw frame."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        undistorted = self.undistorter.undistort_frame(frame)
        
        # Publish
        out_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding='bgr8')
        out_msg.header = msg.header
        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = UndistortionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Deliverable:** ROS 2 node that publishes undistorted stream at 25+ Hz

---

### **TASK 4: Integration with YOLOv8/U-Net Pipeline**
**File:** `src/crack_detection/inference.py` (modify existing)

```python
# In your YOLOv8 detection pipeline:

from insta360_undistortion.undistort import Insta360Undistorter

class DefectDetectionPipeline:
    def __init__(self, model_path: str, calib_dir: str):
        # Load model
        self.model = YOLO(model_path)
        
        # Load undistorter
        K = np.load(f'{calib_dir}/insta360_oner_K.npy')
        D = np.load(f'{calib_dir}/insta360_oner_D.npy')
        self.undistorter = Insta360Undistorter(K, D)
    
    def predict(self, raw_frame: np.ndarray) -> dict:
        """
        Latency breakdown (1920x1080, Jetson Xavier NX):
        
        Undistortion:     12ms  ← cv2.remap + pole crop
        YOLOv8n inference: 18ms  ← GPU TensorRT
        Post-processing:   2ms   ← NMS, parsing
        ────────────────────────
        Total:            32ms   (31.25 Hz real-time ✓)
        """
        # Step 1: Undistort (12ms)
        undistorted = self.undistorter.undistort_frame(raw_frame)
        
        # Step 2: Run YOLOv8 (18ms)
        results = self.model.predict(undistorted, conf=0.6, verbose=False)
        
        # Step 3: Parse results
        detections = {
            'boxes': results[0].boxes.xyxy.cpu().numpy(),
            'confidences': results[0].boxes.conf.cpu().numpy(),
            'class_ids': results[0].boxes.cls.cpu().numpy(),
        }
        
        return detections
```

**Expected mAP Improvement:**
| Condition | mAP | Notes |
|-----------|-----|-------|
| Raw distorted | 78% | Baseline (poor) |
| Undistorted | 91% | +13% ← Your implementation |

---

### **TASK 5: Testing & Validation**
**File:** `tests/test_undistortion.py`

**Test Cases:**

1. **Unit Test: Remap Pre-computation**
   - Verify map_x, map_y are correct shape (H, W)
   - Check values are in valid pixel range [0, W-1], [0, H-1]

2. **Integration Test: End-to-End Undistortion**
   - Load sample Insta360 frame
   - Apply undistortion
   - Verify straight lines remain straight (checkerboard test)

3. **Performance Test: Latency**
   - Measure undistortion time on Jetson Xavier NX
   - Target: <12ms per 1920×1080 frame
   - Log: frame rate, GPU utilization, memory usage

4. **Accuracy Test: Pole Detection**
   - Compare raw vs undistorted pole region detection
   - Target: 40% → 92% recall improvement

```python
def test_undistortion_latency():
    """Measure latency on Jetson Xavier NX."""
    import time
    
    K = np.load('calibration_data/insta360_oner_K.npy')
    D = np.load('calibration_data/insta360_oner_D.npy')
    undistorter = Insta360Undistorter(K, D)
    
    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    times = []
    for _ in range(100):
        start = time.perf_counter()
        undistorted = undistorter.undistort_frame(dummy_frame)
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    mean_latency = np.mean(times)
    std_latency = np.std(times)
    
    print(f"Mean latency: {mean_latency:.2f}ms (±{std_latency:.2f}ms)")
    assert mean_latency < 12.0, f"Latency {mean_latency}ms exceeds 12ms target"
```

**Deliverable:** >90% test pass rate, latency <12ms validated

---

## 📊 Expected Outputs

### Performance (Jetson Xavier NX, 1920×1080)

| Component | Latency | GPU | Status |
|-----------|---------|-----|--------|
| Undistortion (cv2.fisheye) | 12ms | Yes | ✓ Target |
| YOLOv8n inference | 18ms | Yes | Baseline |
| Pole cropping | 1ms | No | Lightweight |
| **Total** | **31ms** | — | **32 Hz ✓** |

### Detection Accuracy

| Metric | Raw | Undistorted | Gain |
|--------|-----|-------------|------|
| mAP @IOU=0.5 | 78% | 91% | +13% |
| Pole detection recall | 40% | 92% | +52% |
| Bbox localization error | ±45px | ±12px | 3.75× |

---

## 🔍 Technical Reference

**Primary Document:** `docs/360_Camera_Undistortion_Methodology.pdf` (13 pages)

**Key Equations:**
- **Fisheye distortion model:** θ_d = r(1 + k₁r² + k₂r⁴ + k₃r⁶ + k₄r⁸)  [Eq 3.3]
- **Bilinear interpolation:** I_u(u,v) = (1-α)(1-β)·I_d(⌊u_d⌋, ⌊v_d⌋) + ... [Eq 5.1]
- **Reprojection RMSE:** √(1/N·Σ||p_measured - p_reprojected||²) [Eq 7.1]

**Calibration (Insta360 ONE R):**
```
K = [[800,   0, 960],
     [  0, 800, 540],
     [  0,   0,   1]]

D = [-0.15, 0.05, -0.02, 0.01]  (Kannala-Breitenstein model)
RMSE = 0.68px (excellent)
```

---

## 🚀 Implementation Checklist

- [ ] **Task 1:** Verify/create calibration files (K.npy, D.npy)
- [ ] **Task 2:** Implement `Insta360Undistorter` class with remap pre-computation
- [ ] **Task 3:** Create ROS 2 subscriber node with `/camera/undistorted` publisher
- [ ] **Task 4:** Integrate into YOLOv8/U-Net pipeline
- [ ] **Task 5:** Unit + integration tests; validate latency <12ms
- [ ] **Task 6:** Document calibration procedure in `CALIBRATION_GUIDE.md`
- [ ] **Task 7:** Push to `insta360-gpu-viz` branch; create PR with benchmarks
- [ ] **Task 8:** Field validation on Jetson Xavier NX (target: 25+ Hz, 91% mAP)

---

## 💬 Notes for Claude Code

1. **Use OpenCV's cv2.fisheye module** — do NOT use cv2.calibrateCamera (perspective lens model). Fisheye requires Kannala-Breitenstein.

2. **Pre-compute remap once** — map_x, map_y are static after calibration. Store in `__init__` to avoid per-frame overhead.

3. **Pole cropping is critical** — Top/bottom 10% of ERP images have extreme distortion. Skip them to avoid YOLOv8 confusion.

4. **Latency budget:** 12ms for undistortion leaves 18ms for YOLOv8n inference = 31ms total = 32 Hz (real-time ✓).

5. **Test on Jetson Xavier NX first** — development laptop CPUs will show misleading latencies. GPU acceleration is essential.

6. **Validate mAP improvement** — Benchmark raw (78%) vs undistorted (91%) on your defect dataset.

7. **ROS 2 integration:** Use `sensor_msgs.Image` + `cv_bridge` for seamless frame passing.

---

## 📝 References

1. OpenCV Fisheye Docs: https://docs.opencv.org/4.5.0/db/d58/group__calib3d__fisheye.html
2. Kannala, J., Breitenstein, M. D. (2006). Generic camera model and calibration for wide-angle lenses. IEEE TPAMI.
3. ROS 2 Camera Calibration: https://github.com/ros-perception/image_pipeline
4. Hartley & Zisserman (2003). Multiple View Geometry in Computer Vision. Cambridge University Press.

---

**Status:** Ready for Claude Code implementation  
**Estimated Effort:** 4-6 hours (calibration → testing)  
**Priority:** 🔴 HIGH (critical for <50mm defect localization accuracy)
