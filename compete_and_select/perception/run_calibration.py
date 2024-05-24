import json
import os

import cv2
import numpy as np
import PIL.Image

from ..apriltag import apriltag

from .collect_calibration_trajectory import calibration_out_dir, waypoints


def print_camera_intrinsics():
    from compete_and_select.perception.rgbd import RGBD

    rgbd = RGBD(num_cameras=1)

    intrinsics = rgbd.cameras[0].intrinsic_matrix
    distortions = rgbd.cameras[0].distortion_coefficients

    print("# Camera parameters for", rgbd.camera_ids[0])
    print("intrinsics = [")
    for i in range(3):
        print(f"    [{intrinsics[i][0]}, {intrinsics[i][1]}, {intrinsics[i][2]}],")
    print("]")
    print("distortions = [" + ", ".join(str(d) for d in distortions) + "]")

def run_calibration():
    detector = apriltag("tag36h11")
    cube_size = 0.04

    image_points = np.zeros((len(waypoints), 2))
    world_points = np.zeros((len(waypoints), 3))
    OK = np.zeros(len(waypoints), dtype=bool)

    for i in range(len(waypoints)):
        # Copy the waypoint to avoid modifying the original
        true_position = np.array(list(waypoints[i]))
        true_position[0] += (cube_size / 2)

        # Load image and convert to grayscale
        img = PIL.Image.open(os.path.join(calibration_out_dir, f"img_{i:02d}.png")).convert("L")
        img = np.array(img)
        detections = detector.detect(img)

        if len(detections) != 1:
            print(f"Error: Expected 1 detection, got {len(detections)}")
            continue
        
        detection = detections[0]
        image_points[i] = detection['center']
        world_points[i] = true_position
        OK[i] = True

    # Filter out apriltags that were not detected.
    if sum(OK) < len(OK):
        print(f"Warning: Only {sum(OK)} of {len(OK)} waypoints were detected.")
    
    image_points = image_points[OK]
    world_points = world_points[OK]

    ret, rvec, translation = cv2.solvePnP(world_points, image_points, intrinsics, distortions)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    translation = translation.T[0]

    return (
        rvec,
        rotation_matrix,
        translation,
    )

# print_camera_intrinsics()

if __name__ == '__main__':
    # Use the known intrinsics and distortion coefficients of the camera.
    # To obtain these, call the "print_camera_intrinsics()" function above.
    # Camera parameters for 000259521012
    intrinsics = [
        [607.46142578125, 0.0, 642.6849365234375],
        [0.0, 607.2731323242188, 364.63818359375],
        [0.0, 0.0, 1.0],
    ]
    distortions = [0.6112022399902344, -3.0245425701141357, 0.0003627542464528233, -6.995165313128382e-05, 1.7275364398956299, 0.4768114984035492, -2.815335988998413, 1.637938380241394]

    intrinsics = np.array(intrinsics)
    distortions = np.array(distortions)

    (rvec, rmat, tvec) = run_calibration()

    with open(os.path.join(os.path.dirname(__file__), "calibration.json"), "w") as f:
        json.dump({
            "rvec": rvec.tolist(),
            "rotation_matrix": rmat.tolist(),
            "translation": tvec.tolist(),
        }, f)
