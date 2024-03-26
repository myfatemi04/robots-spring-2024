import numpy as np

def make_projection(extrinsic, intrinsic, points, orthographic=False):
    camera_translation = extrinsic[:3, 3]
    camera_rotation_matrix = extrinsic[:3, :3]

    # [B, 3] -> ([3, 3] @ [3, B] = [3, B]).T -> [B, 3]
    pose = (camera_rotation_matrix.T @ (points - camera_translation).T).T

    if orthographic:
        # get rid of z (normalize so we can use the translation vector)
        pose[..., 2] = 1

    # [B, 3] -> ([3, 3] @ [3, B] = [3, B]).T -> [B, 3]
    pixel_pose_homogeneous = (intrinsic @ pose.T).T

    # If we do a projection, we divide by the homogeneous coordinate.
    if not orthographic:
        pixel_pose = pixel_pose_homogeneous[..., :2] / pixel_pose_homogeneous[..., [2]]
    else:
        pixel_pose = pixel_pose_homogeneous[..., :2]

    return pixel_pose

def score_to_3d_deflection(score: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    # create a 3D deflection vector.
    # divide by focal lengths.
    score = np.stack([score[..., 0] / intrinsic[0, 0] * 224, score[..., 1] / intrinsic[1, 1] * 224], axis=-1)
    # score[...] = 0
    
    # append a z=0 dimension
    score_3d = np.concatenate([score, np.zeros_like(score[..., 0:1])], axis=-1)
    # score_3d[..., -1] = 1

    # rotate by the extrinsic matrix
    rotation_matrix = extrinsic[:3, :3]
    # ([3, 3] @ ([B, 3].T = [3, B]) = [3, B]).T = [B, 3]
    score_3d = (rotation_matrix @ score_3d.T).T
    
    # print("score_3d:", score_3d)

    # do not add translation; this is a *relative* position!
    return score_3d
