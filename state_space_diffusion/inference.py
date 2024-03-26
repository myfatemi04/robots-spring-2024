import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt

from bilinear_interpolation import bilinear_interpolation
import quaternions as Q
from projections import make_projection, score_to_3d_deflection

def clip_to_scene_bounds(pos, min_bounds, max_bounds):
    pos = torch.max(pos, max_bounds)
    pos = torch.min(pos, min_bounds)
    return pos

def solve_multiview_ivp(
    start_point,
    score_maps,
    quat_maps,
    extrinsics,
    intrinsics,
    projection_types,
    step_sizes,
    scene_bounds,
):
    device = start_point.device
    pos = start_point

    history_3d = [pos.clone()]
    history_2d = []
    history_3d_per_view_scores = []
    history_quats = [np.array([1, 0, 0, 0])]

    T = lambda x: torch.tensor(x, device=device)

    xmin, ymin, zmin, xmax, ymax, zmax = scene_bounds
    max_bounds = torch.tensor([xmax, ymax, zmax], device=device)
    min_bounds = torch.tensor([xmin, ymin, zmin], device=device)

    grid_size = 14
    grid_square_size = 16
    for step in range(len(step_sizes)):
        # calculate score via bilinear interpolation
        max_grid_square_inclusive = grid_size - 1

        # Assumes the intrinsics are correctly rescaled according to image resizing.
        pixel_locations = [
            make_projection(
                extrinsic,
                intrinsic,
                pos.cpu().numpy(),
                projection_type == 'orthographic'
            )
            for extrinsic, intrinsic, projection_type in zip(extrinsics, intrinsics, projection_types)
        ]
        # Find score function from each view. Start from 2D predictions, project to 3D.
        scores_2d = [
            bilinear_interpolation(
                score_map,
                T(pixel_location),
                grid_square_size,
                max_grid_square_inclusive
            )[0]
            for score_map, pixel_location in zip(score_maps, pixel_locations)
        ]
        scores_3d = [
            torch.tensor(
                score_to_3d_deflection(score_2d.cpu().numpy(), extrinsic, intrinsic),
                device=device
            )
            for score_2d, extrinsic, intrinsic in zip(scores_2d, extrinsics, intrinsics)
        ]
        # Find quaternions for each view. Rotate according to rotation matrix of each camera.
        quats_unrotated = [
            bilinear_interpolation(
                quat_map,
                T(pixel_location),
                grid_square_size,
                max_grid_square_inclusive
            )[0].cpu().numpy()
            for quat_map, pixel_location in zip(quat_maps, pixel_locations)
        ]
        quats = [
            Q.compose_quaternions(
                Q.invert_quaternion(Q.rotation_matrix_to_quaternion(extrinsic[:3, :3])),
                quat/np.linalg.norm(quat, axis=-1, keepdims=True)
            )
            for extrinsic, quat in zip(extrinsics, quats_unrotated)
        ]
        quat = np.mean(quats[:1], axis=0)

        score = torch.stack(scores_3d).mean(dim=0)
        
        # print("===")
        # print(np.array([s3d.cpu().numpy() for s3d in scores_3d]))

        # Take a step in the direction of the score
        step_size = step_sizes[step]
        pos += step_size * score
        
        pos = clip_to_scene_bounds(pos, max_bounds, min_bounds)

        history_3d.append(pos.clone())
        history_quats.append(quat)
        history_2d.append(
            torch.stack([torch.tensor(p) for p in pixel_locations], dim=0).to(device)
        )

    history_3d = torch.stack(history_3d, dim=0)
    history_quats = np.stack(history_quats)
    history_2d = torch.stack(history_2d, dim=0)

    return history_3d, history_quats, history_2d

@torch.no_grad()
def infer(model, clip_processor, images, device, image_size=224, grid_size=14):
    """
    Returns score maps (in pixel coordinates) and quaternion maps.
    """
    
    pixel_values = clip_processor(
        images=images,
        return_tensors="pt"
    ).to(device=device).pixel_values # type: ignore

    # langevin dynamics?
    # first, calculate score function. we permute to make axes [batch, x, y, 6]
    maps = model(pixel_values).view(-1, grid_size, grid_size, 6).permute(0, 2, 1, 3)
    pred_direction = maps[..., 0:2].contiguous()
    pred_quat = maps[..., 2:6].contiguous()
        
    pred_direction_px = pred_direction * (224/2)
    
    return (pred_direction_px, pred_quat)

__clip_processor = None
def __get_clip_processor():
    global __clip_processor
    
    if not __clip_processor:
        __clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
    return __clip_processor

SCENE_BOUNDS = [
    -0.3,
    -0.5,
    0.6,
    0.7,
    0.5,
    1.6,
]

def sample_keypoint(model, images, extrinsics, intrinsics, projection_types, start_point, device='cuda'):
    """
    Returns the history of 3D positions and quaternions from the sampling process.
    Also returns the score and quaternion maps used for sampling.
    """
    
    if type(start_point) != torch.Tensor:
        start_point = torch.tensor(start_point, device=device)
    
    score_maps, quat_maps = infer(model, __get_clip_processor(), images, device)
    
    # Use 25 inference steps.
    step_sizes = torch.linspace(0.5, 0.1, 25)
        
    (history_3d, history_quaternions, history_2d_projections) = solve_multiview_ivp(
        start_point,
        score_maps,
        quat_maps,
        extrinsics,
        intrinsics,
        projection_types,
        step_sizes,
        scene_bounds=SCENE_BOUNDS,
    )
    
    return (history_3d, history_quaternions, history_2d_projections, score_maps, quat_maps)
