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
Depth filtering is off by default (the Insta360 has no depth camera), so turn it on for RealSense:

```bash
ros2 launch crack_detection crack_detection.launch.py \
  camera_topic:=/camera/camera/color/image_raw \
  depth_topic:=/camera/camera/depth/image_rect_raw \
  depth_filtering_enabled:=true \
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

The launch file defaults to the front lens (`/insta360/front/image_raw`). To use the compressed
full frame instead (the node decodes the JPEG itself when the topic ends in `/compressed`):

```bash
ros2 launch crack_detection crack_detection.launch.py \
  camera_topic:=/insta360/image_raw/compressed
```

Note that the compressed full frame contains both lenses, so `min_crack_percent` applies to
the whole stacked image and pixel coordinates refer to the 1920x1080 frame.

---

## **View Images with rqt_image_view**

To inspect the visualization output:

```bash
ros2 run rqt_image_view rqt_image_view
```

Then pick **`/crack_detection/visualization compressed`** from the topic dropdown.

The node publishes the visualization twice:

- `/crack_detection/visualization/compressed` (JPEG, ~300 KB per frame): use this one. It runs smoothly in rqt.
- `/crack_detection/visualization` (raw): large frames that best-effort subscribers like rqt mostly drop, so the view lags or freezes.

Do not pass `/crack_detection/visualization/compressed` on the rqt command line: rqt subscribes to it
with the wrong message type and crashes. Select it from the dropdown instead.

The visualization is only built when something subscribes to one of these topics.

---

## **Performance Parameters**

Set in `crack_detection/config/crack_detection_params.yaml`:

| Parameter | Default | Effect |
|---|---|---|
| `use_fp16` | `true` | Half-precision UNet inference on the GPU (about 2x faster, masks match fp32). Ignored on CPU. |
| `viz_scale` | `0.5` | Downscales the published visualization. Use `1.0` for full size. |
| `skip_frames` | `1` | Run inference on every Nth frame. |

With the Insta360 compressed stream on an RTX 2000 Ada, the node keeps up with the camera at about 30 FPS.

---


