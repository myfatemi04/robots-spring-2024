import numpy as np

def parse_intrinsics_and_distortion_coefficients(data, width, height):
    cx, cy, fx, fy, k1, k2, k3, k4, k5, k6, codx, cody, p2, p1 = data
    # fx, fy, cx, fy are scaled by width/height
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
    distcoeffs = np.zeros_like(distcoeffs)
    return intrinsic_matrix, distcoeffs

# Kinect image sizes
WIDTH = 1280
HEIGHT = 720

left_camera_intrinsic_matrix, left_camera_dist_coeffs = parse_intrinsics_and_distortion_coefficients([
    # 0.49961778521537781,
    # 0.50707048177719116,
    0.5,
    0.5,
    0.47331658005714417,
    0.63100129365921021,
    0.43975433707237244, # k1
    -2.7271299362182617, # k2
    1.6047524213790894, # k3
    0.310678094625473, # k4 
    -2.5266492366790771, # k5
    1.5181654691696167, # k6
    0, # codx
    0, # cody
    -0.00012537800648715347, # p2
    0.000460379000287503 # p1
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
