#!/usr/bin/env python3
"""Publish the Insta360 ONE R (USB webcam mode) as ROS 2 image topics.

The camera's MJPEG frames are forwarded untouched as CompressedImage (cheap).
Raw images are only decoded when something subscribes to a raw topic.

Topics:
  /insta360/image_raw/compressed   full frame JPEG (both lenses stacked)
  /insta360/image_raw              full frame, raw bgr8 (decoded on demand)
  /insta360/front/image_raw        bottom half, raw (on demand)
  /insta360/back/image_raw         top half, raw (on demand)
"""
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CompressedImage, Image


class Insta360Publisher(Node):
    def __init__(self):
        super().__init__('insta360_publisher')
        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('frame_id', 'insta360')

        device = self.get_parameter('device').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps = self.get_parameter('fps').value
        self.frame_id = self.get_parameter('frame_id').value

        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Return the raw MJPEG bytes instead of decoding them
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open {device}')
        self.get_logger().info(
            f'Opened {device} at '
            f'{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
            f'{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')

        self.bridge = CvBridge()
        qos = QoSProfile(depth=2)
        self.pub_jpeg = self.create_publisher(
            CompressedImage, 'insta360/image_raw/compressed', qos)
        self.pub = self.create_publisher(Image, 'insta360/image_raw', qos)
        self.pub_front = self.create_publisher(Image, 'insta360/front/image_raw', qos)
        self.pub_back = self.create_publisher(Image, 'insta360/back/image_raw', qos)

        # Grab in a dedicated thread so the camera is read at its own pace
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def publish_raw(self, pub, img, header):
        msg = self.bridge.cv2_to_imgmsg(np.ascontiguousarray(img), encoding='bgr8')
        msg.header = header
        pub.publish(msg)

    def loop(self):
        while self.running and rclpy.ok():
            ok, buf = self.cap.read()
            if not ok:
                self.get_logger().warn('Frame grab failed', throttle_duration_sec=2.0)
                continue

            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.format = 'jpeg'
            msg.data = buf.tobytes()
            self.pub_jpeg.publish(msg)

            raw_pubs = (self.pub, self.pub_front, self.pub_back)
            if not any(p.get_subscription_count() for p in raw_pubs):
                continue
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            h = frame.shape[0] // 2
            if self.pub.get_subscription_count():
                self.publish_raw(self.pub, frame, msg.header)
            if self.pub_back.get_subscription_count():
                self.publish_raw(self.pub_back, frame[:h], msg.header)
            if self.pub_front.get_subscription_count():
                self.publish_raw(self.pub_front, frame[h:], msg.header)

    def destroy_node(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = Insta360Publisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
