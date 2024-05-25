# Camera parameters for 000259521012
import numpy as np

def print_camera_intrinsics():
    from compete_and_select.perception.rgbd import RGBD

    rgbd = RGBD(num_cameras=1)

    for camera_index in range(len(rgbd.cameras)):
        intrinsics = rgbd.cameras[camera_index].intrinsic_matrix
        distortions = rgbd.cameras[camera_index].distortion_coefficients

        print("# Camera parameters for", rgbd.camera_ids[camera_index])
        print("dict(")
        print("\tintrinsics = [")
        for row in range(3):
            print(f"\t\t[{intrinsics[row][0]}, {intrinsics[row][1]}, {intrinsics[row][2]}],")
        print("\t],")
        print("\tdistortions = [" + ", ".join(str(d) for d in distortions) + "]")
        print(")")

camera_params = {
    '000259521012': dict(
        intrinsics = [
            [607.46142578125, 0.0, 642.6849365234375],
            [0.0, 607.2731323242188, 364.63818359375],
            [0.0, 0.0, 1.0],
        ],
        distortions = [0.6112022399902344, -3.0245425701141357, 0.0003627542464528233, -6.995165313128382e-05, 1.7275364398956299, 0.4768114984035492, -2.815335988998413, 1.637938380241394]
    ),
    # Camera parameters for 000243521012
    '000243521012': dict(
        intrinsics = [
            [608.4815673828125, 0.0, 640.407958984375],
            [0.0, 608.4913330078125, 364.266357421875],
            [0.0, 0.0, 1.0],
        ],
        distortions = [0.454001784324646, -3.0351648330688477, 6.399434641934931e-05, -0.00024857273092493415, 1.8521569967269897, 0.3206975758075714, -2.812678098678589, 1.7503421306610107]
    )
}
camera_params = {k: {k2: np.array(v2) for k2, v2 in v.items()} for k, v in camera_params.items()}
