import matplotlib.pyplot as plt
import torch

from projections import make_projection, score_to_3d_deflection
from bilinear_interpolation import bilinear_interpolation

# Visualize the score function in 3D.
def get_3d_score_field(points_3d, score_map, extrinsic, intrinsic):
    if type(points_3d) == torch.Tensor:
        points_3d = points_3d.detach().cpu().numpy()
        
    points_2d = make_projection(extrinsic, intrinsic, points_3d) * (224.0 / 128.0)
    points_2d = torch.tensor(points_2d, device=score_map.device)
    # points_2d[..., 0] = 256 - points_2d[..., 0]
    points_2d[..., 1] = 256 - points_2d[..., 1]
    grid_square_size = 16
    max_grid_square_inclusive = 14 - 1

    score_2d = torch.stack([
        bilinear_interpolation(score_map, points_2d[i], grid_square_size, max_grid_square_inclusive)[0]
        for i in range(len(points_2d))
    ])
    # score_2d[..., 1] *= -1
    score_3d = torch.tensor(
        score_to_3d_deflection(score_2d.cpu().numpy(), extrinsic, intrinsic),
        device=score_map.device
    )
    
    dot_prod_with_camera_direction = score_3d @ torch.tensor(extrinsic[:3, :3].T[2], device='cuda')
    # dot_prod_with_camera_direction = score_3d @ torch.tensor(extrinsic[:-1, [-1]], device='cuda')
    print("abs max:", dot_prod_with_camera_direction.abs().max())
    
    # score_3d = torch.cat([score_2d, torch.zeros_like(score_2d[..., -1:])], dim=-1)
    
    return score_3d

def visualize_3d_score_function(score_maps, extrinsics, intrinsics, ax=None):
    density = 4 + 1
    X = torch.linspace(-0.3, 0.7, density)
    Y = torch.linspace(-0.5, 0.5, density)
    Z = torch.linspace(0.6, 1.6, density)
    
    # grid_xyz_tuple[{0, 1, 2}] are all 3D. Therefore, we resize.
    grid_xyz_tuple = torch.meshgrid(X, Y, Z, indexing='ij')
    points_3d = torch.stack(grid_xyz_tuple, dim=-1).view(-1, 3)
    
    # Find 3D score fields and aggregate them.
    score_fields = [
        get_3d_score_field(points_3d, score_map, extrinsic, intrinsic)
        for (score_map, extrinsic, intrinsic) in zip(score_maps, extrinsics, intrinsics)
    ]
    score_field = torch.stack(score_fields, dim=0).mean(dim=0)
    
    # Display.
    xx, yy, zz = points_3d.cpu().numpy().T
    sx, sy, sz = score_field.cpu().numpy().T
    
    # Don't interfere with existing plots, if they exist.
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_title("Score Function [3D]")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
    # ax.scatter(xx, yy, zz)
    ax.quiver(xx, yy, zz, sx, sy, sz, length=1/density * 0.5)
    # sx *= (1/density)
    # sy *= (1/density)
    # sz *= (1/density)
    # for i in range(len(xx)):
    #     ax.plot([xx[i], xx[i] + sx[i]], [yy[i], yy[i] + sy[i]], [zz[i], zz[i] + sz[i]], c='black')
    
    if ax is None:
        plt.show()
