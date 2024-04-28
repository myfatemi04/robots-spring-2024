import pickle
import sys
from typing import List
import cv2

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
apriltag_object_points[: 0] += 0.025

def enumerate_cameras(num_cameras=2):
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
    def __init__(self, num_cameras=None, camera_ids=None):
        """ camera_ids != None => selects specific cameras for capture """
        k4a_device_map = enumerate_cameras(num_cameras or 2)

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
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
                point_cloud = camera.transform_sensed_points_to_robot_frame(capture.transformed_depth_point_cloud)
                # replace "bad points" with a magic number
                bad_point_mask = (capture.transformed_depth_point_cloud == np.array([0, 0, 0])).all(axis=-1)
                point_cloud[bad_point_mask] = np.array([-10000, -10000, -10000])
                point_clouds.append(point_cloud)
            else:
                point_clouds.append(None)

        return (color_images, point_clouds)
    
    def close(self):
        for camera in self.cameras:
            camera.close()

def get_normal_map(point_cloud_image):
    """
    Given (H, W, 3), a point cloud image, generate surface normals.
    We approximate this as the cross product between derivatives of 3D
    position with respect to image x/y.
    """
    result = np.zeros_like(point_cloud_image)

    # add slight blur
    point_cloud_image_blur = cv2.GaussianBlur(point_cloud_image, ksize=(5,5), sigmaX=2)

    # expected to be right
    horiz_deriv = (point_cloud_image_blur[2:, 1:-1, :] - point_cloud_image_blur[:-2, 1:-1, :]) / 2
    # expected to be down (because y in images is flipped)
    vert_deriv = (point_cloud_image_blur[1:-1, 2:, :] - point_cloud_image_blur[1:-1, :-2, :]) / 2
    horiz_deriv_norm = horiz_deriv / np.linalg.norm(horiz_deriv, axis=-1, keepdims=True)
    vert_deriv_norm = vert_deriv / np.linalg.norm(vert_deriv, axis=-1, keepdims=True)
    # down cross right equals out of the page. normalize
    result[1:-1, 1:-1] = np.cross(vert_deriv_norm, horiz_deriv_norm)

    # get rid of any parts where normal is unknown (any point is (-10000, -10000, -10000))
    bad_points = (point_cloud_image == -10000).all(axis=-1)
    bad_points_orig = np.copy(bad_points)
    bad_points[1:, :] |= bad_points_orig[:-1, :]
    bad_points[:-1, :] |= bad_points_orig[1:, :]
    bad_points[:, 1:] |= bad_points_orig[:, :-1]
    bad_points[:, :-1] |= bad_points_orig[:, 1:]
    result[bad_points] = 0

    return result
