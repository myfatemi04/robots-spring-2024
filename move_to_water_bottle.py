"""
Dummy policy: Move to water bottle.

An idea to quickly follow this one will be to train with RL to avoid obstacles during visual planning.

Will start by just collecting a bunch of images.
"""

from functools import partial
import os
import cv2
import numpy as np
import pyk4a
import PIL.Image

from camera import Camera

class DataCollector:
    def __init__(self, left: Camera, right: Camera, outdir: str):
        self.left_pt = None
        self.right_pt = None
        self.left = left
        self.right = right
        self.left_shown = False
        self.right_shown = False
        self.outdir = outdir
        self.counter = 0

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        cv2.namedWindow("Left")
        cv2.namedWindow("Right")
        cv2.setMouseCallback("Left", partial(self.handle_mouse_event, "Left"))
        cv2.setMouseCallback("Right", partial(self.handle_mouse_event, "Right"))

    def handle_mouse_event(self, side, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if side == 'Left':
                self.left_pt = (x, y)
            elif side == 'Right':
                self.right_pt = (x, y)

    def start(self):
        while True:
            left_capture = self.left.capture()
            right_capture = self.right.capture()

            left_color = np.ascontiguousarray(left_capture.color[:, :, :3])
            right_color = np.ascontiguousarray(right_capture.color[:, :, :3])

            if self.left_pt is not None and self.left_shown:
                go = 'y' == input("Confirm left_pt (y/n):")
                if go:
                    # Save image
                    PIL.Image.fromarray(left_color[:, :, ::-1]).save(
                        os.path.join(self.outdir, f"{self.counter:03d}_left.png")
                    )
                    # Save label
                    with open(
                        os.path.join(self.outdir, f"{self.counter:03d}_left.txt"),
                        "w"
                    ) as f:
                        x, y = self.left_pt
                        f.write(f"{x} {y}")
                    self.counter += 1
                self.left_pt = None
                self.left_shown = False

            if self.right_pt is not None and self.right_shown:
                go = 'y' == input("Confirm right_pt (y/n):")
                if go:
                    # Save image
                    PIL.Image.fromarray(right_color[:, :, ::-1]).save(
                        os.path.join(self.outdir, f"{self.counter:03d}_right.png")
                    )
                    # Save label
                    with open(
                        os.path.join(self.outdir, f"{self.counter:03d}_right.txt"),
                        "w"
                    ) as f:
                        x, y = self.right_pt
                        f.write(f"{x} {y}")
                    self.counter += 1
                self.right_pt = None
                self.right_shown = False

            if self.left_pt is not None:
                cv2.circle(left_color, self.left_pt, 5, (0, 255, 0), 3)
                self.left_shown = True

            if self.right_pt is not None:
                cv2.circle(right_color, self.right_pt, 5, (0, 255, 0), 3)
                self.right_shown = True

            cv2.imshow("Left", left_color)
            cv2.imshow("Right", right_color)

            if cv2.waitKey(1) == ord('q'):
                break

def collect_data():
    k4a_devices = [pyk4a.PyK4A(device_id=i) for i in [0, 1]]
    k4a_device_map = {}
    for device in k4a_devices:
        device.start()
        k4a_device_map[device.serial] = device

    left = Camera(k4a_device_map['000256121012'])
    right = Camera(k4a_device_map['000243521012'])

    capture_number = 1
    collector = DataCollector(left, right, f"human_demonstrations/001_move_to_water_bottle/capture_{capture_number:03d}")
    collector.start()

if __name__ == '__main__':
    collect_data()
