import numpy as np

def voxelize(point_cloud, colors, box_bounds, voxel_size):
    """
    Given a point cloud, generate a voxel grid.
    """
    voxelized_point_cloud = ((point_cloud - box_bounds[0]) / voxel_size).astype(int)
    grid_dims = ((box_bounds[1] - box_bounds[0]) / voxel_size).astype(int)

    # Crop anything that is out-of-bounds.
    mask = ((box_bounds[0] < voxelized_point_cloud) & (voxelized_point_cloud < box_bounds[1])).all(axis=-1)

    # (r, g, b, a)
    voxels = np.zeros((*grid_dims, 4), dtype=float)
    voxels[voxelized_point_cloud[mask], :3] = colors[mask] / 255.0
    voxels[voxelized_point_cloud[mask], -1] = 1

    return voxels


