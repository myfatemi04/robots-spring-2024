import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
)

SCENE_BOUNDS = [
    -0.3,
    -0.5,
    0.6,
    0.7,
    0.5,
    1.6,
]
# [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized

def get_in_bounds_mask(point_cloud, scene_bounds):
    xmin, ymin, zmin, xmax, ymax, zmax = scene_bounds
    return (
        (xmin <= point_cloud[:, 0]) & (point_cloud[:, 0] < xmax) &
        (ymin <= point_cloud[:, 1]) & (point_cloud[:, 1] < ymax) &
        (zmin <= point_cloud[:, 2]) & (point_cloud[:, 2] < zmax)
    )

def get_perspective_transforms():
    """
    Uses a "look-at" view transform.
    """
    elev_azim = {
        "top": (0, 0),
        "front": (90, 0),
        "back": (270, 0),
        "left": (0, 90),
        "right": (0, 270),
    }

    elev = torch.tensor([elev for _, (elev, azim) in elev_azim.items()])
    azim = torch.tensor([azim for _, (elev, azim) in elev_azim.items()])

    up = []
    dist = []
    scale = []
    for view in elev_azim:
        if view in ["left", "right"]:
            up.append((0, 0, 1))
        else:
            up.append((0, 1, 0))

        dist.append(1)
        scale.append((1, 1, 1))

    # "at" kwarg is implicitly the origin.
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, up=up)

    return [R, T, scale]

def get_cube_perspective_rasterizer(img_size, radius, device):
    R, T, scale = get_perspective_transforms()
    rasterizer = PointsRasterizer(
        cameras=FoVOrthographicCameras(
            R=R,
            T=T,
            znear=0.01,
            scale_xyz=scale,
            device=device,
        ),
        raster_settings=PointsRasterizationSettings(
            image_size=img_size,
            # Radius is in NDC units, which is also "Normalized Device Coordinates".
            # NDC coordinates define the unit cube of points that are allowed in a scene.
            # It is a construct of projective geometry, in which points are confined to:
            # (x, y, z) \in [-1, 1] \times [-1, 1] \times [z_near, z_far]
            radius=radius,
            # How many points are saved per pixel. For example, if several points happen to
            # coincide.
            points_per_pixel=8,
            # Only affects the speed of the forward pass.
            bin_size=0,
        )
    )
    return rasterizer
