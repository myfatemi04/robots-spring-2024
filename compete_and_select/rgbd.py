import pickle
import sys
from typing import List

sys.path.insert(0, "../")
from camera import Camera
import apriltag
sys.path.remove("../")

import numpy as np
import pyk4a

apriltag_detector = apriltag.apriltag("tag36h11")
apriltag_object_points = np.array([
    # order: left bottom, right bottom, right top, left top
    [1, -1/2, 0],
    [1, +1/2, 0],
    [0, +1/2, 0],
    [0, -1/2, 0],
]).astype(np.float32) * 0.1778

def enumerate_cameras():
    num_cameras = 2
    if pyk4a.connected_device_count() < num_cameras:
        print(f"Error: Not enough K4A devices connected (<{num_cameras}).")
        exit(1)

    k4a_devices = [pyk4a.PyK4A(device_id=i) for i in range(num_cameras)]
    k4a_device_map = {}
    for device in k4a_devices:
        device.start()
        k4a_device_map[device.serial] = device

    return k4a_device_map

def _color(capture: pyk4a.PyK4ACapture):
    return np.ascontiguousarray(capture.color[..., :3][..., ::-1])

class RGBD:
    def __init__(self, camera_ids=None):
        """ camera_ids != None => selects specific cameras for capture """
        k4a_device_map = enumerate_cameras()

        if camera_ids is None:
            camera_ids = list(k4a_device_map.keys())

        self.camera_ids = camera_ids

        k4as = []
        for camera_id in k4a_device_map.keys():
            if camera_id in camera_ids:
                k4as.append(k4a_device_map[camera_id])
            else:
                k4a_device_map[camera_id].stop()

        self.cameras = [Camera(k4a) for k4a in k4as]

    def try_calibrate(self, camera_index, image):
        camera = self.cameras[camera_index]
        image_gray = image.mean(axis=-1).astype(np.uint8)
        try:
            detections = apriltag_detector.detect(image_gray)
        except RuntimeError:
            print("AprilTag runtime error.", file=sys.stderr)
            detections = []
        if len(detections) == 1:
            detection = detections[0]
            apriltag_image_points = detection['lb-rb-rt-lt']
            camera.infer_extrinsics(apriltag_image_points, apriltag_object_points)
            return True
        return False

    def capture(self):
        """
        Creates a capture. Adds an RGBD point cloud to each detection, if AprilTags were found (or if an existing calibration exists)
        """
        
        # Attempt to calibrate uncalibrated cameras.
        captures = [camera.capture() for camera in self.cameras]

        color_images = []
        point_clouds = []

        for i, (camera, capture) in enumerate(zip(self.cameras, captures)):
            color = _color(captures[i])
            color_images.append(color)
            
            if camera.extrinsic_matrix is None:
                success = self.try_calibrate(i, color)
                if success:
                    print("Calibrated camera", self.camera_ids[i])

            if camera.extrinsic_matrix is not None:
                point_clouds.append(
                    camera.transform_sensed_points_to_robot_frame(capture.transformed_depth_point_cloud)
                )
            else:
                point_clouds.append(None)

        return (color_images, point_clouds)
    
    def close(self):
        for camera in self.cameras:
            camera.close()
