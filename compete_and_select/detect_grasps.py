import cv2
import matplotlib.pyplot as plt
import numpy as np

from .util.set_axes_equal import set_axes_equal
from .voxelize import voxelize


def get_normal(window):
    # we use finite differences method.
    # window is a 3x3 matrix.
    dx = (window[2, 1] - window[0, 1]) / 2
    dy = (window[1, 2] - window[1, 0]) / 2
    normal = np.array([dx, dy, 1])
    return normal / np.linalg.norm(normal)

def smoothen_window(window):
    # median blur.
    # alternative: SMA.
    # ^^ convolve2d(window_min_z, np.ones((3, 3)) / 9, mode='same', boundary='symm')
    return cv2.medianBlur(window.astype(np.float32), 3) # type: ignore

def detect_grasps_z_axis(voxel_occupancy, voxel_size, gripper_width, max_alpha, hop_size=2, window_size=2):
    """
    Detects vertical grasps in a voxel grid, assuming a parallel-jaw gripper.
    Uses z as the gripper direction.

    we create several windows over the point cloud
    then for each window we find the minimum and maximum z-values; this tells us the contact point
    additionally we calculate the normal vector at those voxels
    finally, we can check if the grasp is force-closure by looking at the friction cone
    voxelization is just to reduce the number of points in our point cloud to save processing
    """
    max_z = np.zeros((voxel_occupancy.shape[0], voxel_occupancy.shape[1])) - 1
    min_z = np.zeros((voxel_occupancy.shape[0], voxel_occupancy.shape[1])) + 10000

    for z in range(voxel_occupancy.shape[2]):
        mask = voxel_occupancy[:, :, z] > 0
        if not np.any(mask):
            continue
        max_z[mask] = np.maximum(max_z[mask], z)
        min_z[mask] = np.minimum(min_z[mask], z)

    grasp_locations = []

    for x in range(window_size, voxel_occupancy.shape[0] - window_size, hop_size):
        for y in range(window_size, voxel_occupancy.shape[1] - window_size, hop_size):
            window_min_z = min_z[x - window_size:x + window_size, y - window_size:y + window_size]
            window_max_z = max_z[x - window_size:x + window_size, y - window_size:y + window_size]
            zmin = np.min(window_min_z) - 1
            zmax = np.max(window_max_z) + 1

            if np.max(window_min_z) == 10000 or np.min(window_max_z) == -1:
                continue

            # get normal vector at this point.
            # smoothen the window.
            window_min_z = smoothen_window(window_min_z)
            window_max_z = smoothen_window(window_max_z)

            # with this smoothed window, calculate the normal (which is the finite differences of the z-values)
            lower_norm = get_normal(window_min_z[window_size - 1:window_size + 2, window_size - 1:window_size + 2])
            upper_norm = get_normal(window_max_z[window_size - 1:window_size + 2, window_size - 1:window_size + 2])

            # contact direction is vertical
            alpha_lower = np.degrees(np.arccos(lower_norm[2]))
            alpha_upper = np.degrees(np.arccos(upper_norm[2]))
            alpha_lower = min(alpha_lower, 180 - alpha_lower)
            alpha_upper = min(alpha_upper, 180 - alpha_upper)

            # if x == 16 and y == 8:
            #     print(alpha_lower, alpha_upper)
            #     print(lower_norm, upper_norm)
            #     print(window_min_z)
            #     print(window_max_z)

            # then, calculate alpha
            # finally, see if it's inside or outside the friction cone
            # will just assume that if |alpha| < 15deg, we're fine
            contact_distance = (zmax - zmin) * voxel_size
            if max(np.abs(alpha_lower), np.abs(alpha_upper)) < max_alpha and contact_distance < gripper_width:
                grasp_locations.append((x, y, zmin, zmax, alpha_lower, alpha_upper))

    return grasp_locations

def detect_grasps(object_points, object_point_colors, voxel_size=0.005, min_points_in_voxel=2, gripper_width=0.08, max_alpha=15, hop_size=1, window_size=2, top_k_per_angle=5, show_rotated_voxel_grids=False):
    grasps = []

    # take only the top 0.035m of object points to ensure that the gripper can reach
    gripper_depth = 0.035
    top_z = max(object_points[:, 2])
    chop_object_mask = object_points[:, 2] >= top_z - gripper_depth
    object_points_chopped = object_points[chop_object_mask]
    object_point_colors_chopped = object_point_colors[chop_object_mask]

    for i in range(8):
        rotate_angle = np.pi / 8 * i
        z_inv = np.array([np.cos(rotate_angle), np.sin(rotate_angle), 0])
        x_inv = np.array([np.cos(rotate_angle - np.pi/2), np.sin(rotate_angle - np.pi/2), 0])
        y_inv = np.cross(z_inv, x_inv)
        rotation_matrix = np.array([x_inv, y_inv, z_inv])

        # apply rotation matrix to points
        rotated_object_points = object_points_chopped @ rotation_matrix.T
        lower_bound_, upper_bound_ = np.min(rotated_object_points, axis=0), np.max(rotated_object_points, axis=0)
        voxels_ = voxelize(rotated_object_points, object_point_colors_chopped, (lower_bound_, upper_bound_), voxel_size)

        voxel_occupancy_ = (voxels_[:, :, :, -1] >= min_points_in_voxel)
        grasps_voxelized = detect_grasps_z_axis(voxel_occupancy_, voxel_size, gripper_width, max_alpha, hop_size, window_size)
        # translate these into the original frame.
        # these are in (x, y, zmin, zmax) format.
        grasps_from_this = []
        for (x, y, zmin, zmax, alpha_lower, alpha_upper) in grasps_voxelized:
            worst_alpha = max(abs(alpha_lower), abs(alpha_upper))
            # print(f"Grasp at x={x} y={y} worst_alpha={worst_alpha:.2f}")
            start = (np.array([x, y, zmin]) * voxel_size + lower_bound_) @ rotation_matrix
            end = (np.array([x, y, zmax]) * voxel_size + lower_bound_) @ rotation_matrix
            grasps_from_this.append((worst_alpha, start, end))

        # select top grasps by force closure
        grasps_from_this.sort(key=lambda x: x[0])
        grasps.extend(grasps_from_this[:top_k_per_angle])

        if show_rotated_voxel_grids:
            fig = plt.figure()
            ax: plt.Axes = fig.add_subplot(projection='3d')
            ax.set_title(f"Rotation Angle: $\\frac{{{i}\\pi}}{{8}}$")

            voxel_color_ = voxels_.copy()
            voxel_color_[..., -1] = 1.0
            ax.voxels(voxel_occupancy_, facecolors=voxel_color_, edgecolor=(1, 1, 1, 0.1)) # type: ignore
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z') # type: ignore
            set_axes_equal(ax)
            plt.show()

    return grasps
