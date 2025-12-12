# Crack Detection UNet-Pix2Pix ROS2 Package

This repository contains a ROS2 package for crack detection using a UNet+Pix2Pix model. It includes integration with RealSense cameras, visualization in RViz, and image viewing via `rqt_image_view`.
---

## **Setup**

Make sure you have:

- ROS2 installed
- RealSense2 ROS package ([`realsense2_camera`](https://github.com/IntelRealSense/realsense-ros)) installed  
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

Start the crack detection visualization with namespace support and zoom settings:

```bash
ros2 launch crack_detection crack_detection.launch.py \
  camera_topic:=/camera/camera/color/image_raw \
  depth_topic:=/camera/camera/depth/image_rect_raw \
  zoom_enabled:=true \
  zoom_factor:=1.5
```
---

## **View Images with rqt_image_view**

To inspect the visualization output:

```bash
ros2 run rqt_image_view rqt_image_view /crack_detection/visualization
```
---


