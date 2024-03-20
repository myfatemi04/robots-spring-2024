import torch

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

class VoxelRenderer:
    def __init__(
        self,
        scene_bounds,
        voxel_size,
        background_color,
        device,
    ):
        self.scene_bounds = scene_bounds
        self.mins = torch.tensor(self.scene_bounds[:3], device=device)
        self.maxs = torch.tensor(self.scene_bounds[3:], device=device)
        self.voxel_size = voxel_size
        self.background_color = background_color
        self.device = device

    def _discretize_point(self, xyz: torch.Tensor):
        return (
            # voxel_size represents a camera intrinsic matrix
            self.voxel_size * (xyz - self.mins)
            # assume coordinates follow the same aspect ratio
            # / (self.maxs - self.mins) # scale
        ).to(torch.long) # discretize

    def point_location_on_images(self, xyz: torch.Tensor):
        x, y, z = self._discretize_point(xyz)
        # in (x, y) order.
        return ((y.item(), z.item()), (x.item(), z.item()), (x.item(), y.item()))

    def convert_2d_to_3d(self, zy: torch.Tensor, zx: torch.Tensor, xy: torch.Tensor):
        d, h = zy.shape
        w = zx.shape[1]
        return (
            zy.permute(1, 0).unsqueeze(0).repeat(w, 1, 1) +
            zx.permute(1, 0).unsqueeze(1).repeat(1, h, 1) +
            xy.unsqueeze(2).repeat(1, 1, d)
        )
    
    def voxelize(self, point_cloud: torch.Tensor, color: torch.Tensor):
        mask = get_in_bounds_mask(point_cloud, self.scene_bounds)
        point_cloud = point_cloud[mask]
        color = color[mask]
        scaled_discretized = self._discretize_point(point_cloud)
        color_3d = torch.zeros((self.voxel_size, self.voxel_size, self.voxel_size, 3), dtype=color.dtype, device=self.device)
        mask_3d = torch.zeros((self.voxel_size,) * 3, dtype=torch.int, device=self.device)
        X = scaled_discretized[:, 0]
        Y = scaled_discretized[:, 1]
        Z = scaled_discretized[:, 2]
        color_3d[X, Y, Z] = color
        mask_3d[X, Y, Z] = 1
        return (color_3d, mask_3d)
    
    def render_voxels_orthographically(self, color_3d: torch.Tensor, mask_3d: torch.Tensor):
        x_size, y_size, z_size, _ = color_3d.shape

        # iterate over each slice and render it
        yz_image = torch.ones((z_size, y_size, 3), dtype=color_3d.dtype, device=self.device)
        yz_image[:] = self.background_color
        for x in range(x_size):
            mask_slice = mask_3d[x, :, :].T.unsqueeze(-1)
            color_slice = color_3d[x, :, :].permute(1, 0, 2)

            yz_image = yz_image * (1 - mask_slice) + color_slice
            
        xz_image = torch.ones((z_size, x_size, 3), dtype=color_3d.dtype, device=self.device)
        xz_image[:] = self.background_color
        for y in range(y_size):
            # [x, z] -> [z, x]
            mask_slice = mask_3d[:, y, :].T.unsqueeze(-1)
            color_slice = color_3d[:, y, :].permute(1, 0, 2)

            xz_image = xz_image * (1 - mask_slice) + color_slice
        
        xy_image = torch.ones((y_size, x_size, 3), dtype=color_3d.dtype, device=self.device)
        xy_image[:] = self.background_color
        for z in range(z_size):
            # [x, y] -> [y, x]
            mask_slice = mask_3d[:, :, z].T.unsqueeze(-1)
            color_slice = color_3d[:, :, z].permute(1, 0, 2)
            
            xy_image = xy_image * (1 - mask_slice) + color_slice

        return (yz_image, xz_image, xy_image)
    
    def render_point_cloud(self, point_cloud: torch.Tensor, color: torch.Tensor):
        color_3d, mask_3d = self.voxelize(point_cloud, color)
        return self.render_voxels_orthographically(color_3d, mask_3d)
    
    def __call__(self, point_cloud: torch.Tensor, color: torch.Tensor):
        return self.render_point_cloud(point_cloud, color)
