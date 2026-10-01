# Run Summary: Insta360 ONE R + Crack Detection

Measured on 2026-10-01 on an NVIDIA RTX 2000 Ada (16 GB) with a 24-core CPU and 31.5 GB RAM, ROS 2 Jazzy (Fast DDS).
Rates come from `ros2 topic hz`, CPU from `pidstat`, and GPU from `nvidia-smi`. Values marked *estimated* are worked out from the code, not measured.

## Pipeline

```
Insta360 ONE R (USB, MJPG) → insta360_publisher → /insta360/image_raw/compressed
   → crack_detection_node (UNet, GPU) → /crack_detection/visualization (+ /compressed)
```

## Camera: `insta360_publisher`

| | |
|---|---|
| Device | `/dev/video0`, MJPG |
| Resolution | **1920×1080** (both lenses stacked: back on top, front on the bottom) |
| Requested rate | 30 FPS |
| Measured `/insta360/image_raw/compressed` | **~30.0 Hz** |
| Per-lens topics (`/insta360/front/image_raw`, `/insta360/back/image_raw`) | 1920×540 each, decoded only when subscribed |

## Detector: `crack_detection_node`

| | |
|---|---|
| Input topic | `/insta360/image_raw/compressed` (full 1920×1080 frame, both lenses) |
| Model | UNet, **384×384** input, FP16, GPU |
| Tiling / Pix2Pix refinement | Off / off |
| Frames processed | Every frame (`skip_frames: 1`) |
| Measured detection rate | **~30 FPS** (log lines about 33 ms apart) |
| Detection rule | Crack ≥ 2.0% of the frame, in 4 of the last 5 frames. Hysteresis switches the alert off below 1.0%, with a 0.5 s minimum duration. |
| Depth filtering | Off (the Insta360 has no depth sensor) |

## Visualization output

| Topic | Resolution | Size per frame | Measured rate |
|---|---|---|---|
| `/crack_detection/visualization` (raw) | ~2880×660 *(estimated)* | ~5–6 MB | **30.2 Hz** with the Fast DDS profile (≈1.7 Hz without it) |
| `/crack_detection/visualization/compressed` | same, JPEG | ~200–300 KB | **~29.9 Hz** |

With `viz_scale: 0.5`, each 1920×1080 frame becomes 960×540. The image puts three of those side by side (original, heatmap and overlay) with a 120 px info strip on top.
At 30 Hz the raw topic moves roughly 150–170 MB/s. See the "Fast DDS profile" section of the [README](../README.md) for why the raw topic needs `config/fastdds_large_images.xml`.

## Resource use

| | |
|---|---|
| `crack_detection_node` CPU | ~1.2 cores (118%: 93% user, 25% system), the main limit |
| `insta360_publisher` CPU | ~8% |
| GPU utilization | ~37–39% |
| GPU memory (detector) | ~520 MB of 16 GB |
| GPU temperature / power | 56 °C, 34 W |
| System | ~93% idle, 6.5 / 31.5 GB RAM used |

## Bottom line

The full pipeline keeps up with the camera's 30 FPS from start to finish, with plenty of GPU left over.
The detector is limited by CPU (one Python process), not GPU. If you raise the load with tiling, Pix2Pix or `viz_scale: 1.0`, CPU will run out first.
