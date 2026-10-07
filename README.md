# Crack Detection UNet-Pix2Pix ROS2 Package

This repository contains ROS2 packages for crack detection using a UNet+Pix2Pix model. It includes integration with RealSense cameras and the Insta360 ONE R, visualization in RViz, and image viewing via `rqt_image_view`.
---

## **Setup**

Make sure you have:

- ROS2 installed
- RealSense2 ROS package ([`realsense2_camera`](https://github.com/IntelRealSense/realsense-ros)) installed (only for RealSense)
- Git LFS is installed before cloning. Large model files are handled via Git LFS.
- Your Python environment configured for the crack detection model
---

## **Install Git LFS if not installed**
```bash
# Linux (Ubuntu/Debian)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt update
sudo apt install git-lfs
git lfs install
```
---
## **Repository Layout**

The repository holds two ROS2 packages. Clone it into your workspace `src/` and colcon builds both:

```
crack-detection-unet-pix2pix/
├── crack_detection/   # crack detection node, launch file, config, models
└── insta360_ros/      # Insta360 ONE R camera publisher
```
---
## **Clone the Repository**

```bash
cd ~/ros2_ws/src
git clone https://github.com/Yongann-v/crack-detection-unet-pix2pix.git
cd crack-detection-unet-pix2pix
git lfs pull
```
--- 

## **Install RealSense2 ROS Package if not installed**

The RealSense2 ROS package allows ROS2 to interface with Intel RealSense cameras.

### Step 1 — Install dependencies
```bash
sudo apt-get update
sudo apt-get install -y \
  git cmake build-essential \
  libusb-1.0-0-dev pkg-config
```

### Step 2 — Clone the repository
```bash
cd ~/ros2_ws/src
git clone https://github.com/IntelRealSense/realsense-ros.git
```
For ROS2 Humble or later, checkout the ros2 branch:
```bash
cd realsense-ros
git checkout ros2
```
### Step 3 — Build the workspace and source your workspace
```bash
cd ~/ros2_ws
colcon build --symlink-install
source ~/ros2_ws/install/setup.bash
```
---

## **Launch RealSense Camera**

Start the RealSense camera node:

```bash
ros2 launch realsense2_camera rs_launch.py
```

## **Launch Crack Detection Visualization in RViz**

Start the crack detection visualization with namespace support and zoom settings.
Depth filtering is off by default (the Insta360 has no depth camera), so turn it on for RealSense.
Insta360 lens undistortion is on by default, so turn it off for RealSense:

```bash
ros2 launch crack_detection crack_detection.launch.py \
  camera_topic:=/camera/camera/color/image_raw \
  depth_topic:=/camera/camera/depth/image_rect_raw \
  depth_filtering_enabled:=true \
  undistort_enabled:=false \
  zoom_enabled:=true \
  zoom_factor:=1.5
```
---

## **Run with the Insta360 ONE R**

The `insta360_ros` package in this repo publishes the Insta360 ONE R (USB webcam mode) as ROS 2 image topics.
It is built together with `crack_detection`, so no extra install is needed:

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

Connect the camera over USB in webcam mode and check it shows up (your user needs to be in the `video` group):

```bash
v4l2-ctl --list-devices   # should list "Insta360 One R" -> /dev/video0
```

### Step 1 — Start the camera publisher

```bash
ros2 run insta360_ros insta360_publisher
# Optional: --ros-args -p device:=/dev/video0 -p width:=1920 -p height:=1080 -p fps:=30.0
```

It publishes:

| Topic | Type | Content |
|---|---|---|
| `/insta360/image_raw/compressed` | `CompressedImage` | Full frame JPEG, both lenses stacked (always published, cheap) |
| `/insta360/image_raw` | `Image` | Full frame, decoded only when subscribed |
| `/insta360/front/image_raw` | `Image` | Bottom half (front lens), decoded only when subscribed |
| `/insta360/back/image_raw` | `Image` | Top half (back lens), decoded only when subscribed |

### Step 2 — Start crack detection

The launch file defaults to the front lens (`/insta360/front/image_raw`), undistorted with the
front-lens calibration (see [Insta360 Lens Undistortion](#insta360-lens-undistortion)).

Back lens:

```bash
ros2 launch crack_detection crack_detection.launch.py \
  camera_topic:=/insta360/back/image_raw \
  calibration_file:=calibration_data/insta360_oner_back.yaml
```

To use the compressed full frame instead (the node decodes the JPEG itself when the topic ends in `/compressed`):

```bash
ros2 launch crack_detection crack_detection.launch.py \
  camera_topic:=/insta360/image_raw/compressed
```

With undistortion on (the default), the node crops the stacked frame to the calibrated lens
before undistorting, so detection runs on that lens only. With `undistort_enabled:=false` the
whole stacked frame is used: `min_crack_percent` then applies to both lenses together and pixel
coordinates refer to the 1920x1080 frame.

---

## **Insta360 Lens Undistortion**

In webcam mode the Insta360 ONE R sends both lenses as two 1920x540 strips (back lens on top,
front lens on the bottom). The firmware partly flattens them, but barrel distortion remains,
so cracks near the left and right edges look bent. The detection node removes it with a
per-lens fisheye calibration before zoom and inference, so masks, crack percentage and crack
centers are all in undistorted image coordinates.

| Parameter | Default | Effect |
|---|---|---|
| `undistort_enabled` | `true` | Undistort frames before inference. Set `false` for RealSense or other cameras. |
| `calibration_file` | `calibration_data/insta360_oner_front.yaml` | Lens calibration (relative to the package share directory). |
| `undistort_balance` | `0.5` | `0` crops to valid pixels (loses ~40% of the view), `1` keeps the full view with large black corners. |

Calibrations in `crack_detection/calibration_data/` (9x6 board, 20 mm squares, fisheye model):

| Lens | File | Reprojection error | Images |
|---|---|---|---|
| Front | `insta360_oner_front.yaml` | 0.32 px | 79 |
| Back | `insta360_oner_back.yaml` | 0.195 px | 82 |

Undistortion costs about 0.45 ms per frame, and the node still runs at about 30 Hz in UNet fast mode.
If the frame size does not match the calibration, the node logs an error and skips the frame.

### Recalibrate a lens

Stop `insta360_publisher` first (the capture tool opens `/dev/video0` directly).

```bash
# 1. Print a checkerboard at 100% scale and measure one square
ros2 run crack_detection capture_calibration --make-board checkerboard.png --board 9x6 --square-mm 20

# 2. Capture 40+ images (space = save, a = auto-save, q = quit). Cover the corners and edges.
#    PYTHONNOUSERSITE=1 is needed if pip's opencv-python-headless is installed in ~/.local
PYTHONNOUSERSITE=1 ros2 run crack_detection capture_calibration --lens front --board 9x6

# 3. Calibrate (writes the YAML and a before/after preview image)
ros2 run crack_detection calibrate_insta360 --images calib_images/front --lens front \
  --board 9x6 --square-mm 20 --out src/crack-detection-unet-pix2pix/crack_detection/calibration_data

# 4. Rebuild so the new calibration is installed
colcon build --packages-select crack_detection --symlink-install
```

Aim for a reprojection error under 0.5 px. Tests for the undistorter are in
`Agent_md/test_undistortion_template.py` (`python3 -m pytest Agent_md/test_undistortion_template.py -v`).

---

## **View Images with rqt_image_view**

To open the visualization output directly:

```bash
ros2 run rqt_image_view rqt_image_view /crack_detection/visualization
```

Or run `ros2 run rqt_image_view rqt_image_view` with no argument and pick a topic from the dropdown.

The node publishes the visualization twice:

| Topic | Size per frame | Notes |
|---|---|---|
| `/crack_detection/visualization` | ~5 MB (raw BGR) | Full quality, ~30 Hz. Needs the Fast DDS profile below. |
| `/crack_detection/visualization/compressed` | ~200–300 KB (JPEG) | Lighter. Select **`/crack_detection/visualization (compressed)`** in the dropdown. |

Do not pass a `.../compressed` topic on the rqt command line: rqt subscribes to it
with the wrong message type and crashes. Select it from the dropdown instead.
`camera_topic:=...` is a crack detection node argument, not an rqt one.

The visualization is only built when something subscribes to one of these topics.

### Fast DDS profile for the raw visualization

Fast DDS's default shared-memory segment is 512 KB, so the ~5 MB raw frames get split into
many fragments, and best-effort subscribers like rqt drop almost all of them (measured: ~1.7 Hz
instead of 30 Hz). `crack_detection/config/fastdds_large_images.xml` raises the segment to 64 MB
and keeps UDPv4, so topics from the robot over the network still work.

- **With the launch file:** applied to the detector node automatically.
- **With `ros2 run`:** export it in that terminal before starting the node:

  ```bash
  export FASTRTPS_DEFAULT_PROFILES_FILE=$(ros2 pkg prefix crack_detection)/share/crack_detection/config/fastdds_large_images.xml
  ros2 run crack_detection crack_detection_node --ros-args -p camera_topic:=/insta360/image_raw/compressed
  ```

Only the publisher needs the profile. Viewers like rqt work without it.

---

## **Performance Parameters**

Set in `crack_detection/config/crack_detection_params.yaml`:

| Parameter | Default | Effect |
|---|---|---|
| `use_fp16` | `true` | Half-precision UNet inference on the GPU (about 2x faster, masks match fp32). Ignored on CPU. |
| `viz_scale` | `0.5` | Downscales the published visualization. Use `1.0` for full size. |
| `skip_frames` | `1` | Run inference on every Nth frame. |

With the Insta360 compressed stream on an RTX 2000 Ada, the node keeps up with the camera at about 30 FPS.

Measured resource use in that setup (Insta360 compressed input, raw visualization subscribed):

| | Usage |
|---|---|
| `crack_detection_node` CPU | ~1.2 cores (~118%) |
| `insta360_publisher` CPU | ~8% |
| GPU utilization | ~37–39% |
| GPU memory (detector) | ~520 MB |

The node is CPU-bound (single Python process), not GPU-bound. See [docs/run_summary.md](docs/run_summary.md)
for the full run summary (resolutions, frame rates, bandwidth). To monitor:

```bash
watch -n1 nvidia-smi
top -p $(pgrep -d, -f "crack_detection_node|insta360")
```

---


