import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyk4a
import cv2
import numpy as np
import apriltag
import os
import sys
sys.path.insert(0, "./hand_object_detector")
import detect_hands_pipeline
os.chdir("hand_object_detector")
detect_hands_pipeline.initialize()
os.chdir("..")

def parse_intrinsics_and_distortion_coefficients(data, width, height):
    cx, cy, fx, fy, k1, k2, k3, k4, k5, k6, codx, cody, p2, p1 = data
    fx *= width
    cx *= width
    fy *= height
    cy *= height
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ])
    distcoeffs = np.array([k1, k2, p1, p2]) # , k3, k4, k5, k6])
    return intrinsic_matrix, distcoeffs

# Load camera with the default config
# Device names:
# 000243521012 [right camera]
# 000256121012 [left camera]

print("Num. Connected Devices:", pyk4a.connected_device_count())

k4a_left = pyk4a.PyK4A(device_id=1)# 256121012)
k4a_left.start()

k4a_right = pyk4a.PyK4A(device_id=0)# 243521012)
k4a_right.start()

apriltag_detector = apriltag.apriltag("tag36h11")
apriltag_object_points = np.array([
    # order: left bottom, right bottom, right top, left top
    [1, -1/2, 0],
    [1, +1/2, 0],
    [0, +1/2, 0],
    [0, -1/2, 0],
]).astype(np.float32) * 0.1778

apriltag_mini_detector = apriltag.apriltag("tagStandard41h12")

demo_cube_object_points = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
]).astype(np.float32)

# if not os.path.exists("left_calibration.json"):
#     k4a_left.save_calibration_json("left_calibration.json")
#     k4a_right.save_calibration_json("right_calibration.json")

WIDTH = 1280
HEIGHT = 720

# For plotting 3D object detections
fig = plt.figure()
object_detection_rendering_ax = fig.add_subplot(projection='3d', computed_zorder=False)

# fx, fy, cx, cy are given in relative coordinates.
# pyk4a.Calibration()

use_k4a_intrinsics = True
if use_k4a_intrinsics:
    right_camera_intrinsic_matrix = k4a_right.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    right_camera_dist_coeffs = k4a_right.calibration.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)
    left_camera_intrinsic_matrix = k4a_left.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    left_camera_dist_coeffs = k4a_left.calibration.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)
else:
    left_camera_intrinsic_matrix, left_camera_dist_coeffs = parse_intrinsics_and_distortion_coefficients([
        0.49961778521537781,
        0.50707048177719116,
        0.47331658005714417,
        0.63100129365921021,
        0.43975433707237244,
        -2.7271299362182617,
        1.6047524213790894,
        0.310678094625473,
        -2.5266492366790771,
        1.5181654691696167,
        0,
        0,
        -0.00012537800648715347,
        0.000460379000287503
    ], WIDTH, HEIGHT)

    right_camera_intrinsic_matrix, right_camera_dist_coeffs = parse_intrinsics_and_distortion_coefficients([
        0.50070935487747192,
        0.50496494770050049,
        0.47537624835968018,
        0.63384515047073364,
        0.454001784324646,
        -3.0351648330688477,
        1.8521569967269897,
        0.32069757580757141,
        -2.8126780986785889,
        1.7503421306610107,
        0,
        0,
        -0.00024857273092493415,
        6.3994346419349313E-5
    ], WIDTH, HEIGHT)

do_hand_detection = True

left_extrinsic_matrix = None
right_extrinsic_matrix = None

try:
    while True:
        left_capture = k4a_left.get_capture()
        right_capture = k4a_right.get_capture()

        left_camera_hand_center = None
        right_camera_hand_center = None
        
        left_color = np.ascontiguousarray(left_capture.color[:, :, :3])
        if left_extrinsic_matrix is None:
            left_gray = cv2.cvtColor(left_color, cv2.COLOR_RGB2GRAY)
            try:
                tags = apriltag_detector.detect(left_gray)
            except RuntimeError:
                print("left_gray had threading error")
            for tag in tags:
                points = tag['lb-rb-rt-lt'].astype(np.float32)
                cv2.drawContours(left_color, [points.astype(int)], 0, (0, 255, 0), 3)
                # Get extrinsics
                # https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
                ret, rvec, tvec = cv2.solvePnP(apriltag_object_points, points, left_camera_intrinsic_matrix, left_camera_dist_coeffs)
                left_rotation_matrix, _ = cv2.Rodrigues(rvec)
                left_translation = tvec
                left_extrinsic_matrix = np.concatenate((left_rotation_matrix, left_translation), axis=1)

                # Plot corners of cube ({0, 1}, {0, 1}, {0, 1})
                # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
                for (multiplier, color) in [(0.1, (255, 0, 0)), (0.5, (0, 255, 0)), (0.7, (0, 0, 255)), (1.0, (255, 255, 255))][::-1]:
                    points, _jacobian = cv2.projectPoints(demo_cube_object_points * multiplier, rvec, tvec, left_camera_intrinsic_matrix, left_camera_dist_coeffs)
                    points = points[:, 0, :]
                    for point in points:
                        cv2.circle(left_color, point.astype(int), 15, color, 3)
                    # draw line to connect them
                    # [0, 0, 0],
                    # [0, 1, 0],
                    # [1, 0, 0],
                    # [1, 1, 0],
                    # [0, 0, 1],
                    # [0, 1, 1],
                    # [1, 0, 1],
                    # [1, 1, 1],
                    for (a, b) in [
                        (0, 1),
                        (0, 2),
                        (0, 4),
                        (1, 3),
                        (1, 5),
                        (2, 3),
                        (2, 6),
                        (3, 7),
                        (4, 5),
                        (4, 6),
                        (5, 7),
                        (6, 7),
                    ]:
                        cv2.line(left_color, points[a].astype(int), points[b].astype(int), color, 3)

                # To create a rotation matrix, we use Rodrigues
                # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
                # rot, _jacobian = cv2.Rodrigues(rvecs)

        if do_hand_detection:
            hand_bbxs, hand_scores = detect_hands_pipeline.detect(left_color)
            for hand_bbox in hand_bbxs:
                hand_bbox = np.round(hand_bbox).astype(int)
                x1, y1, x2, y2 = hand_bbox.astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                left_camera_hand_center = (center_x, center_y)
                # cv2.rectangle(right_color, (hand_bbox[0], max(0, hand_bbox[1]-30)), (hand_bbox[0]+62, max(0, hand_bbox[1]-30)+30), (255, 255, 255), 4)
                cv2.rectangle(left_color, (x1, y1), (x2, y2), (255, 255, 255), 4)
                cv2.circle(left_color, (center_x, center_y), 15, (0, 0, 255), 3)

        right_color = np.ascontiguousarray(right_capture.color[:, :, :3])
        if right_extrinsic_matrix is None:
            right_gray = cv2.cvtColor(right_color, cv2.COLOR_RGB2GRAY)
            try:
                tags = apriltag_detector.detect(right_gray)
            except RuntimeError:
                print("right_gray had threading error")

            for tag in tags:
                points = tag['lb-rb-rt-lt'].astype(np.float32)
                cv2.drawContours(right_color, [points.astype(int)], 0, (0, 255, 0), 3)
                # Get extrinsics
                # https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
                ret, rvec, tvec = cv2.solvePnP(apriltag_object_points, points, right_camera_intrinsic_matrix, right_camera_dist_coeffs)
                right_rotation_matrix, _ = cv2.Rodrigues(rvec)
                right_translation = tvec
                right_extrinsic_matrix = np.concatenate((right_rotation_matrix, right_translation), axis=1)

                # Plot corners of cube ({0, 1}, {0, 1}, {0, 1}) at different scales
                # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
                for (multiplier, color) in [(0.1, (255, 0, 0)), (0.5, (0, 255, 0)), (0.7, (0, 0, 255)), (1.0, (255, 255, 255))][::-1]:
                    demo_cube_object_points[:, 1] *= -1
                    points, _jacobian = cv2.projectPoints(demo_cube_object_points * multiplier, rvec, tvec, right_camera_intrinsic_matrix, right_camera_dist_coeffs)
                    points = points[:, 0, :]
                    demo_cube_object_points[:, 1] *= -1
                    for point in points:
                        cv2.circle(right_color, point.astype(int), 15, color, 3)
                    # draw line to connect them
                    # [0, 0, 0],
                    # [0, 1, 0],
                    # [1, 0, 0],
                    # [1, 1, 0],
                    # [0, 0, 1],
                    # [0, 1, 1],
                    # [1, 0, 1],
                    # [1, 1, 1],
                    for (a, b) in [
                        (0, 1),
                        (0, 2),
                        (0, 4),
                        (1, 3),
                        (1, 5),
                        (2, 3),
                        (2, 6),
                        (3, 7),
                        (4, 5),
                        (4, 6),
                        (5, 7),
                        (6, 7),
                    ]:
                        cv2.line(right_color, points[a].astype(int), points[b].astype(int), color, 3)

        if do_hand_detection:
            hand_bbxs, hand_scores = detect_hands_pipeline.detect(right_color)
            for hand_bbox in hand_bbxs:
                hand_bbox = np.round(hand_bbox).astype(int)
                x1, y1, x2, y2 = hand_bbox.astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                right_camera_hand_center = (center_x, center_y)
                # cv2.rectangle(right_color, (hand_bbox[0], max(0, hand_bbox[1]-30)), (hand_bbox[0]+62, max(0, hand_bbox[1]-30)+30), (255, 255, 255), 4)
                cv2.rectangle(right_color, (x1, y1), (x2, y2), (255, 255, 255), 4)
                cv2.circle(right_color, (center_x, center_y), 15, (0, 0, 255), 3)

        # Draw detections for cubes
        # for detection in apriltag_mini_detector.detect(left_gray):
        #     points = tag['lb-rb-rt-lt'].astype(np.float32)
        #     cv2.drawContours(left_color, [points.astype(int)], 0, (0, 128, 255), 3)

        # for detection in apriltag_mini_detector.detect(right_gray):
        #     points = tag['lb-rb-rt-lt'].astype(np.float32)
        #     cv2.drawContours(right_color, [points.astype(int)], 0, (0, 128, 255), 3)

        """
        Depth capture is (576, 640)
        Image capture is (720, 1280)
        """

        # Detect hands with `hand_object_detector`

        """
        Now, let's say we want to triangulate some points.
        We can do this through OpenCV's triangulatePoints method.
        Note that this requires undistorted points; we will estimate
        these through cv2.undistortPoints().
        
        https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
        """

        # Scale depth inversely
        # ld = left_capture.depth
        # ld = ld/ld.max()
        # rd = right_capture.depth
        # rd = rd/rd.max()
        # cv2.imshow('left_depth', ld)
        # cv2.imshow('right_depth', rd)

        cv2.imshow('left', left_color)
        cv2.imshow('right', right_color)

        if left_camera_hand_center is not None \
            and right_camera_hand_center is not None \
            and left_extrinsic_matrix is not None \
            and right_extrinsic_matrix is not None:
            # undistort points
            # note: this undoes the intrinsic matrix as well
            undistorted_left = cv2.undistortPoints(np.array([left_camera_hand_center]).astype(np.float32), left_camera_intrinsic_matrix, left_camera_dist_coeffs)
            undistorted_right = cv2.undistortPoints(np.array([right_camera_hand_center]).astype(np.float32), right_camera_intrinsic_matrix, right_camera_dist_coeffs)
            print(undistorted_left[0], undistorted_right[0])
            # therefore here, you must pretend that the intrinsic matrix is the identity
            # the *_extrinsic_matrix parameters are supposed to be camera matrices (i.e. intrinsic @ extrinsic)
            # but we just use extrinsic to pretend that the intrinsic matrix is the identity
            triangulated_homogenous = cv2.triangulatePoints(left_extrinsic_matrix, right_extrinsic_matrix, undistorted_left, undistorted_right)
            triangulated_homogenous = triangulated_homogenous.T
            print(triangulated_homogenous)
            triangulated = triangulated_homogenous[:, :3] / triangulated_homogenous[:, -1]
            print("Triangulated:", triangulated)
            x, y, z = triangulated[0]
            object_detection_rendering_ax.clear()
            object_detection_rendering_ax.view_init(elev=30, azim=0, roll=0)
            object_detection_rendering_ax.set_title("World Coordinates and Detections")
            # Add background
            vertices = [
                np.array([
                    (0, -1, -0.1),
                    (0,  1, -0.1),
                    (1,  1, -0.1),
                    (1, -1, -0.1),
                ]),
                np.array([
                    (0, -1, -0.1),
                    (0,  1, -0.1),
                    (0,  1, 1),
                    (0, -1, 1),
                ]),
            ]
            object_detection_rendering_ax.add_collection3d(Poly3DCollection(vertices, color=(0.2, 0.2, 0.2, 1.0)))
            object_detection_rendering_ax.scatter(x, y, z, label='Hand', s=25, c=(0, 1, 0, 1.0))
            object_detection_rendering_ax.scatter(*apriltag_object_points.T, label='AprilTag points', s=5, c='r')
            object_detection_rendering_ax.set_xlabel("X (m)")
            object_detection_rendering_ax.set_ylabel("Y (m)")
            object_detection_rendering_ax.set_zlabel("Z (m)")
            object_detection_rendering_ax.set_xlim(-1, 1)
            object_detection_rendering_ax.set_ylim(-1, 1)
            object_detection_rendering_ax.set_zlim(-1, 1)
            object_detection_rendering_ax.legend()
            plt.pause(0.1)

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    k4a_left.close()
    k4a_right.close()
