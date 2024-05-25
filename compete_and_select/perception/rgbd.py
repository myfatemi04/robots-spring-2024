import json
import os
import sys

import cv2
import numpy as np
import PIL.Image

from .camera import Camera

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
import apriltag

sys.path.pop(0)

def enumerate_cameras(min_cameras=2):
    import pyk4a  # type: ignore

    if pyk4a.connected_device_count() < min_cameras:
        print(f"Error: Not enough K4A devices connected (<{min_cameras}).")
        exit(1)

    k4a_devices = [pyk4a.PyK4A(device_id=i) for i in range(pyk4a.connected_device_count())]
    k4a_device_map = {}
    for device in k4a_devices:
        device.start()
        k4a_device_map[device.serial] = device

    return k4a_device_map

def _color(capture):
    return np.ascontiguousarray(capture.color[..., :3][..., ::-1])

_preset_configurations = {
    'diagonal': [
        ('000259521012', 'front_left'),
        ('000243521012', 'back_right')
    ],
    'front_only_color': [('000259521012', None)]
}

class RGBD:
    @staticmethod
    def autoload(camera_id_to_calibration_preset):
        if type(camera_id_to_calibration_preset) is str:
            camera_id_to_calibration_preset = _preset_configurations[camera_id_to_calibration_preset]

        # Cameras are loaded in a deterministic order according to their ID.
        rgbd = RGBD(camera_ids=[cid for cid, cpt in camera_id_to_calibration_preset])

        for i, (camera_id, extrinsic_id) in enumerate(camera_id_to_calibration_preset):
            if extrinsic_id is None:
                continue

            with open(os.path.join(os.path.dirname(__file__), f"extrinsics/{extrinsic_id}_camera.json")) as f:
                extrinsics = json.load(f)
            extrinsics = {k: np.array(v) for k, v in extrinsics.items()}

            rgbd.cameras[i].rvec_tvec = (extrinsics['rvec'], extrinsics['translation'])
            rgbd.cameras[i].extrinsic_matrix = np.concatenate([
                extrinsics['rotation_matrix'],
                extrinsics['translation'][:, np.newaxis]
            ], axis=1)
            
        return rgbd

    def __init__(self, num_cameras=None, camera_ids=None, auto_calibrate=False):
        """ camera_ids != None => selects specific cameras for capture """
        if camera_ids is not None:
            if num_cameras is not None:
                assert len(camera_ids) == num_cameras, "num_cameras must match the number of camera_ids, if both are specified"
            
            num_cameras = len(camera_ids)
        
        k4a_device_map = enumerate_cameras(num_cameras or 2)

        if camera_ids is None:
            camera_ids = list(k4a_device_map.keys())

        self.camera_ids = camera_ids
        self.auto_calibrate = auto_calibrate

        k4as = []
        remaining = set(k4a_device_map.keys())
        for camera_id in camera_ids:
            k4as.append(k4a_device_map[camera_id])
            remaining.remove(camera_id)
        
        for camera_id in remaining:
            k4a_device_map[camera_id].stop()

        self.cameras = [Camera(k4a) for k4a in k4as]

        self.apriltag_detector = apriltag.apriltag("tag36h11") # type: ignore
        apriltag_object_points = np.array([
            # order: left bottom, right bottom, right top, left top
            [1, -1/2, 0],
            [1, +1/2, 0],
            [0, +1/2, 0],
            [0, -1/2, 0],
        ]).astype(np.float32)
        apriltag_object_points[:, 0] += 1
        apriltag_object_points *= 0.1778
        apriltag_object_points[: 0] += 0.025
        self.apriltag_object_points = apriltag_object_points

    def try_calibrate(self, camera_index, image):
        camera = self.cameras[camera_index]
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # image_gray = cv2.equalizeHist(image_gray)
        # plt.title("Calibration Image")
        # plt.imshow(image_gray)
        # plt.show()
        try:
            detections = self.apriltag_detector.detect(image_gray)
        except RuntimeError as e:
            print("AprilTag runtime error:", e, file=sys.stderr)
            detections = []
        # print(detections)
        if len(detections) == 1:
            print("Detected AprilTag", camera_index)
            detection = detections[0]
            apriltag_image_points = detection['lb-rb-rt-lt']
            camera.infer_extrinsics(apriltag_image_points, self.apriltag_object_points)
            return True
        return False

    def capture(self, return_pil=False):
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
            
            if camera.extrinsic_matrix is None and self.auto_calibrate:
                success = self.try_calibrate(i, color)
                if success:
                    print("Calibrated camera", self.camera_ids[i])

            if camera.extrinsic_matrix is not None:
                point_cloud = camera.transform_sensed_points_to_robot_frame(capture.transformed_depth_point_cloud)
                assert point_cloud is not None
                # replace "bad points" with a magic number
                bad_point_mask = (capture.transformed_depth_point_cloud == np.array([0, 0, 0])).all(axis=-1)
                point_cloud[bad_point_mask] = np.array([-10000, -10000, -10000])
                point_clouds.append(point_cloud)
            else:
                point_clouds.append(None)

        if return_pil:
            color_images = [PIL.Image.fromarray(rgb) for rgb in color_images]

        return (color_images, point_clouds)
    
    def close(self):
        for camera in self.cameras:
            camera.close()

    def __del__(self):
        self.close()

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

def obtain_calibration(rgbd: RGBD, tracker, checkpoint_path="calibrations.pkl", force_retry=False):
    import pickle

    import matplotlib.pyplot as plt
    
    if not os.path.exists(checkpoint_path) or force_retry:
        has_pcd = False
        while not has_pcd:
            # uses a threading.Event to wait for next frame
            if tracker is not None:
                (rgbs, pcds, _) = tracker.next()
                for i in range(len(rgbs)):
                    tracker.rgbd.try_calibrate(i, rgbs[i])
            else:
                rgbs, pcds = rgbd.capture()
                for i in range(len(rgbs)):
                    rgbd.try_calibrate(i, rgbs[i])
                    
            has_pcd = all(pcd is not None for pcd in pcds)
            plt.title("Camera 1")
            plt.imshow(rgbs[1])
            plt.pause(0.05)

        # save calibration
        calibrations = [rgbd.cameras[0].extrinsic_matrix, rgbd.cameras[1].extrinsic_matrix]
        with open(checkpoint_path, "wb") as f:
            pickle.dump(calibrations, f)
        
        print("Saved calibrations.")
    else:
        with open(checkpoint_path, "rb") as f:
            calibrations = pickle.load(f)
            for i, calibration in enumerate(calibrations[:len(rgbd.cameras)]):
                rgbd.cameras[i].extrinsic_matrix = calibration

        print("Restored calibrations.")


