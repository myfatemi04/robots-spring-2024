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
    distcoeffs = np.array([k1, k2, p1, p2])
    return intrinsic_matrix, distcoeffs

# Load camera with the default config
# Device names:
# 000243521012 [right camera]
# 000256121012 [left camera]

print("Num. Connected Devices:", pyk4a.connected_device_count())

k4a_left = pyk4a.PyK4A(device_id=0)# 256121012)
k4a_left.start()

k4a_right = pyk4a.PyK4A(device_id=1)# 243521012)
k4a_right.start()

apriltag_detector = apriltag.apriltag("tag36h11")
apriltag_object_points = np.array([
    # order: left bottom, right bottom, right top, left top
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
]).astype(np.float32)

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

# fx, fy, cx, cy are given in relative coordinates.
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

try:
    while True:
        left_capture = k4a_left.get_capture()
        right_capture = k4a_right.get_capture()

        left_color = np.ascontiguousarray(left_capture.color[:, :, :3])
        left_gray = cv2.cvtColor(left_color, cv2.COLOR_RGB2GRAY)
        tags = apriltag_detector.detect(left_gray)
        hand_bbxs, hand_scores = detect_hands_pipeline.detect(left_color)
        for hand_bbx in hand_bbxs:
            x, y, w, h = hand_bbx.astype(int)
            # cv2.rectangle(left_color, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.circle(left_color, (x, y), 15, (0, 0, 255), 3)
        for tag in tags:
            points = tag['lb-rb-rt-lt'].astype(np.float32)
            cv2.drawContours(left_color, [points.astype(int)], 0, (0, 255, 0), 3)
            # Get extrinsics
            # https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
            ret, rvec, tvec = cv2.solvePnP(apriltag_object_points, points, left_camera_intrinsic_matrix, left_camera_dist_coeffs)

            # Plot corners of cube ({0, 1}, {0, 1}, {0, 1})
            # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
            points, _idk = cv2.projectPoints(demo_cube_object_points, rvec, tvec, left_camera_intrinsic_matrix, left_camera_dist_coeffs)
            points = points[:, 0, :]
            for point in points:
                cv2.circle(left_color, point.astype(int), 15, (255, 0, 0), 3)

            # To create a rotation matrix, we use Rodrigues
            # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
            # rot, _jacobian = cv2.Rodrigues(rvecs)

        right_color = np.ascontiguousarray(right_capture.color[:, :, :3])
        right_gray = cv2.cvtColor(right_color, cv2.COLOR_RGB2GRAY)
        tags = apriltag_detector.detect(right_gray)
        hand_bbxs, hand_scores = detect_hands_pipeline.detect(right_color)
        for hand_bbx in hand_bbxs:
            x, y, w, h = hand_bbx.astype(int)
            # cv2.rectangle(left_color, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.circle(right_color, (x, y), 15, (0, 0, 255), 3)
        for tag in tags:
            points = tag['lb-rb-rt-lt'].astype(np.float32)
            cv2.drawContours(right_color, [points.astype(int)], 0, (0, 255, 0), 3)
            # Get extrinsics
            # https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
            ret, rvec, tvec = cv2.solvePnP(apriltag_object_points, points, right_camera_intrinsic_matrix, right_camera_dist_coeffs)

            # Plot corners of cube ({0, 1}, {0, 1}, {0, 1})
            # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
            points, _idk = cv2.projectPoints(demo_cube_object_points, rvec, tvec, right_camera_intrinsic_matrix, right_camera_dist_coeffs)
            points = points[:, 0, :]
            for point in points:
                cv2.circle(right_color, point.astype(int), 15, (255, 0, 0), 3)

            # To create a rotation matrix, we use Rodrigues
            # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
            # rot, _jacobian = cv2.Rodrigues(rvecs)
            pass

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
        ld = left_capture.depth
        ld = ld/ld.max()
        rd = right_capture.depth
        rd = rd/rd.max()

        cv2.imshow('left', left_color)
        cv2.imshow('right', right_color)
        cv2.imshow('left_depth', ld)
        cv2.imshow('right_depth', rd)

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    k4a_left.close()
    k4a_right.close()
