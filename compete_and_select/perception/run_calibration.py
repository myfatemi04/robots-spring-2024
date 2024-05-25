import json
import os

import cv2
import numpy as np
import PIL.Image
from matplotlib import pyplot as plt

from ..apriltag import apriltag
from .camera_params import camera_params
from .collect_calibration_trajectory import waypoints as W


def run_calibration(preset, intrinsics, distortions):
    # decimate=2 by default, but for small AprilTags, this
    # can make detection more difficult. So we set it to 1.
    detector = apriltag("tag36h11", decimate=1)
    cube_size = 0.04

    calibration_out_dir = os.path.join(os.path.dirname(__file__), f"calibration_images_{preset}")

    try:
        with open(os.path.join(calibration_out_dir, "eef_poses.json"), "r") as f:
            waypoints = json.load(f)
    except:
        print("Warning: Could not load eef_poses.json. Using the 'ground truth' waypoints, which may be non-representative")
        print("of the actual robot position in the images.")
        waypoints = W[preset]

    image_points = np.zeros((len(waypoints), 2))
    world_points = np.zeros((len(waypoints), 3))
    OK = np.zeros(len(waypoints), dtype=bool)

    for i in range(len(waypoints)):
        # Copy the waypoint to avoid modifying the original
        true_position = np.array(list(waypoints[i]))

        # Compensation for AprilTag not being exactly at the center of end-effector.
        if 'front' in preset:
            true_position[0] += (cube_size / 2)
        else:
            true_position[0] -= (cube_size / 2)

        # The weird Z offset thing
        true_position[2] -= 0.1034

        # Load image and convert to grayscale
        img_path = os.path.join(calibration_out_dir, f"img_{i:02d}.png")
        if not os.path.exists(img_path):
            print("Skipping image", img_path, "because it does not exist.")
            continue

        img = PIL.Image.open(img_path).convert("L")
        img = np.array(img)
        img = cv2.equalizeHist(img)
        detections = detector.detect(img)

        if len(detections) != 1:
            print(f"Error: Expected 1 detection, got {len(detections)}")
            continue
        
        detection = detections[0]
        image_points[i] = detection['center']
        world_points[i] = true_position

        # plt.imshow(img, cmap='gray')
        # plt.scatter(*image_points[i], color='red')
        # plt.show()

        OK[i] = True

    # Filter out apriltags that were not detected.
    if sum(OK) < len(OK):
        print(f"Warning: Only {sum(OK)} of {len(OK)} waypoints were detected.")
    
    image_points = image_points[OK]
    world_points = world_points[OK]

    ret, rvec, translation = cv2.solvePnP(world_points[:9], image_points[:9], intrinsics, distortions)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    translation = translation.T[0]

    reprojected_image_points = cv2.projectPoints(world_points, rvec, translation, intrinsics, distortions)[0]
    reprojected_image_points = reprojected_image_points[:, 0, :]

    # Plot the reprojected points
    plt.title("Reprojected points")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.scatter(image_points[:, 0], image_points[:, 1], color='red', label='Detected')
    plt.scatter(reprojected_image_points[:, 0], reprojected_image_points[:, 1], color='blue', label='Reprojected')
    plt.legend()
    plt.show()


    return (
        rvec,
        rotation_matrix,
        translation,
    )

# print_camera_intrinsics()

if __name__ == '__main__':
    # Use the known intrinsics and distortion coefficients of the camera.
    # To obtain these, call the "print_camera_intrinsics()" function above.


    (rvec, rmat, tvec) = run_calibration('back_right', **camera_params['000243521012'])
    with open(os.path.join(os.path.dirname(__file__), "extrinsics/back_right_camera.json"), "w") as f:
        json.dump({
            "rvec": rvec.tolist(),
            "rotation_matrix": rmat.tolist(),
            "translation": tvec.tolist(),
        }, f)

    (rvec, rmat, tvec) = run_calibration('front_left', **camera_params['000259521012'])
    with open(os.path.join(os.path.dirname(__file__), "extrinsics/front_left_camera.json"), "w") as f:
        json.dump({
            "rvec": rvec.tolist(),
            "rotation_matrix": rmat.tolist(),
            "translation": tvec.tolist(),
        }, f)

