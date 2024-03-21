### v8: 

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import quaternions as Q
import torch
import torch.utils.data
import tqdm
import transformers
from demo_to_state_action_pairs import (CAMERAS, create_labels_v3,
                                        create_torch_dataset_v3,
                                        make_projection, prepare_image)
from get_data import get_demos
from model_architectures import VisualPlanDiffuserV7
from voxel_renderer_slow import SCENE_BOUNDS, VoxelRenderer


def get_diffusion_schedule(fix_alpha_scaling=True, device="cuda"):
    betas = torch.linspace(0.002, 0.5, 10, device=device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    if fix_alpha_scaling:
        # scale so alphas_cumprod is 0 at final time
        alphas_cumprod_scaled = (alphas_cumprod - alphas_cumprod[-1]) * (alphas_cumprod[0] / (alphas_cumprod[0] - alphas_cumprod[-1]))
        # reverse-engineer the correct alphas.
        alphas_cumprod_starting = torch.tensor([1.0, *alphas_cumprod_scaled])
        alphas_cumprod_ending = torch.tensor([*alphas_cumprod_scaled, 0.0])
        alphas_scaled = alphas_cumprod_ending / alphas_cumprod_starting
        betas = 1 - alphas_scaled
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod_scaled)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod_scaled)

    return (betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

def image_coordinates_to_noise_coordinates(position: torch.Tensor, image_scaling: float, center: float):
    return (position - center) / image_scaling

def noise_coordinates_to_image_coordinates(noise: torch.Tensor, image_scaling: float, center: float):
    return noise * image_scaling + center

def get_noisy_position(position: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor):
    return torch.sqrt(1 - sigma ** 2) * position + sigma * noise

image_scaling = 112
center = 112
NI = partial(noise_coordinates_to_image_coordinates, image_scaling=image_scaling, center=center)
IN = partial(image_coordinates_to_noise_coordinates, image_scaling=image_scaling, center=center)
NP = lambda x: x.detach().cpu().numpy()

def scaled_arrows(image, pixel_scaled_positions, pred_direction, true_direction):
    if type(image) == torch.Tensor:
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
    plt.imshow(image, origin="lower")

    pred_arrow_ends = NI(pixel_scaled_positions)
    true_arrow_ends = NI(pixel_scaled_positions)

    plt.quiver(
        NP(pred_arrow_ends[:, 0]),
        NP(pred_arrow_ends[:, 1]),
        NP(pred_direction[:, 0]) * image_scaling,
        NP(pred_direction[:, 1]) * image_scaling,
        color='red',
        label='Predicted'
    )
    plt.quiver(
        NP(true_arrow_ends[:, 0]),
        NP(true_arrow_ends[:, 1]),
        NP(true_direction[:, 0]) * image_scaling,
        NP(true_direction[:, 1]) * image_scaling,
        color='blue',
        label='True'
    )

def train(demos, epochs=20):
    device = torch.device('cuda')

    clip = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device) # type: ignore

    freeze_first_n_layers = 0
    for i in range(freeze_first_n_layers):
        clip.vision_model.encoder.layers[i].requires_grad_(False)
    print("Freezing first", freeze_first_n_layers, "layers out of", len(clip.vision_model.encoder.layers))

    clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16") # type: ignore

    model = VisualPlanDiffuserV7(clip).to(device)

    for name, param in model.named_parameters():
        # check if MLP
        if "linear" in name and ('.weight' in name or '.bias' in name):
            param.data.fill_(0)

    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    dataset = create_torch_dataset_v3(demos, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    grid_size = 14

    for epoch in range(epochs):
        for (image, position, quat) in (pbar := tqdm.tqdm(dataloader)):
            pixel_values = clip_processor(images=image, return_tensors="pt", do_rescale=False).to(device=device).pixel_values # type: ignore

            scaled_target_position = image_coordinates_to_noise_coordinates(position, image_scaling, center)
            pixel_scaled_positions = torch.zeros((grid_size, grid_size, 2), device=device)
            pixel_scaled_positions[..., 0] = (0.5 + torch.arange(grid_size, device=device).view(grid_size, 1).expand(grid_size, grid_size)) * 2 / grid_size - 1
            pixel_scaled_positions[..., 1] = (0.5 + torch.arange(grid_size, device=device).view(1, grid_size).expand(grid_size, grid_size)) * 2 / grid_size - 1
            pixel_scaled_positions = pixel_scaled_positions.unsqueeze(0).expand(position.shape[0], -1, -1, -1)

            distances = (pixel_scaled_positions - scaled_target_position.view(-1, 1, 1, 2)).norm(dim=-1, keepdim=True)
            quat_loss_weighting = torch.exp(-distances / 0.1)
            quat_loss_weighting = quat_loss_weighting / quat_loss_weighting.sum(dim=(1, 2, 3), keepdim=True)
            # unsqueeze spatial dims
            quat = quat.view(-1, 1, 1, 4)

            true_direction = (scaled_target_position.view(-1, 1, 1, 2) - pixel_scaled_positions)

            # (batch, token_y, token_x, 2) -> (batch, token_x, token_y, 2)
            pred = model(pixel_values).view(-1, grid_size, grid_size, 6).permute(0, 2, 1, 3)
            pred_direction = pred[..., 0:2].contiguous()
            pred_quat = pred[..., 2:6].contiguous()

            score_loss = (pred_direction - true_direction).pow(2).mean()

            quat_loss_coeff = 10
            quat_loss = quat_loss_coeff * ((pred_quat - quat).pow(2) * quat_loss_weighting).mean()
            loss = score_loss + quat_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            pbar.set_postfix({"loss": loss.item(), "quat_loss": quat_loss.item(), "score_loss": score_loss.item()})

        # Visualize the vector field
        if (epoch + 1) % 10 == 0:
            plt.clf()
            plt.title("Score Function")
            # Project from 2D memory layout to serial layout
            scaled_arrows(
                image[0],
                pixel_scaled_positions[0].view(-1, 2),
                pred_direction[0].view(-1, 2),
                true_direction[0].view(-1, 2),
            )
            plt.legend()
            plt.savefig(f"diffusion_{epoch}.png")

    torch.save(model.state_dict(), "diffusion_model.pt")

def bilinear_interpolation(features: torch.Tensor, xy: torch.Tensor, grid_square_size: float, max_grid_square_inclusive: int):
    x = xy[..., 0]
    y = xy[..., 1]

    lower_ind = torch.floor((xy - (grid_square_size / 2)) / grid_square_size).to(torch.long)
    lower_ind[lower_ind < 0] = 0
    lower_ind[lower_ind > (max_grid_square_inclusive - 1)] = max_grid_square_inclusive - 1
    lower_x_ind = lower_ind[..., 0]
    lower_y_ind = lower_ind[..., 1]
    upper_x_ind = lower_x_ind + 1
    upper_y_ind = lower_y_ind + 1
    lower_x = (lower_x_ind + 0.5) * grid_square_size
    lower_y = (lower_y_ind + 0.5) * grid_square_size
    upper_x = (upper_x_ind + 0.5) * grid_square_size
    upper_y = (upper_y_ind + 0.5) * grid_square_size

    # interpolated_value = 1/((upper_x - lower_x) * (upper_y - lower_y)) * (
    interpolated_value = 1/(grid_square_size ** 2) * (
        features[lower_x_ind, lower_y_ind] * (upper_x - x) * (upper_y - y) +
        features[upper_x_ind, lower_y_ind] * (x - lower_x) * (upper_y - y) +
        features[lower_x_ind, upper_y_ind] * (upper_x - x) * (y - lower_y) +
        features[upper_x_ind, upper_y_ind] * (x - lower_x) * (y - lower_y)
    )

    return interpolated_value, (lower_x_ind, lower_y_ind)

def score_to_3d_deflection(score: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    # create a 3D deflection vector.
    # divide by focal lengths.
    # score = np.stack([score[..., 0] / intrinsic[0, 0], score[..., 1] / intrinsic[1, 1]], axis=-1)
    score = np.stack([score[..., 0] / intrinsic[0, 0] * 224, score[..., 1] / intrinsic[1, 1] * 224], axis=-1)
    # append a z=0 dimension
    score_3d = np.concatenate([score, np.zeros_like(score[..., 0:1])], axis=-1)

    # rotate by the extrinsic matrix
    rotation_matrix = extrinsic[:3, :3]
    score_3d = rotation_matrix @ score_3d

    # do not add translation; this is a *relative* position!
    return score_3d

# Let's try sampling from the model. Given multiple views, let's try to use this score function + Langevin dynamics
# to sample a 3D position. Ideally, because we're using multiple views, we'll be able to extract 3D positions using
# ONLY 2D training data!
# At some point, we will want to add noise conditioning.
def sample(model, images, extrinsics, intrinsics, start_point, device="cuda"):
    clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16") # type: ignore
    pixel_values = clip_processor(
        images=images,
        return_tensors="pt",
        do_rescale=False
    ).to(device=device).pixel_values # type: ignore

    # langevin dynamics
    # first, calculate score function. we permute to make axes [batch, x, y, 6]
    with torch.no_grad():
        maps = model(pixel_values).view(-1, 14, 14, 6).permute(0, 2, 1, 3)
        score_maps = maps[..., 0:2].contiguous()
        quat_maps = maps[..., 2:6].contiguous()
    pos = torch.tensor(start_point, device=device)

    history_3d = [pos]
    history_2d = []
    # will store this in numpy
    history_quats = [np.array([1, 0, 0, 0])]

    T = lambda x: torch.tensor(x, device=device)

    xmin, ymin, zmin, xmax, ymax, zmax = SCENE_BOUNDS

    n_steps = 10
    step_sizes = torch.linspace(0.2, 0.1, n_steps, device=device)
    grid_size = 14
    grid_square_size = 16
    for step in range(n_steps):
        print("Running sampling step", step)
        # calculate score via bilinear interpolation
        max_grid_square_inclusive = grid_size - 1

        pixel_locations = [make_projection(extrinsic, intrinsic, pos.cpu().numpy()) for extrinsic, intrinsic in zip(extrinsics, intrinsics)]
        scores_unrotated = [
            bilinear_interpolation(score_map, (T(pixel_location)), grid_square_size, max_grid_square_inclusive)[0]
            for score_map, pixel_location in zip(score_maps, pixel_locations)
        ]
        # calculate 3d deflection vectors according to scores predicted in pixel space.
        # account for focal length, etc. (i.e., pixel space -> world space length conversion)
        scores = [
            torch.tensor(
                score_to_3d_deflection(score_2d.cpu().numpy(), extrinsic, intrinsic),
                device=device
            )[:1]
            for score_2d, extrinsic, intrinsic in zip(scores_unrotated, extrinsics, intrinsics)
        ]
        quats_unrotated = [
            bilinear_interpolation(quat_map, (T(pixel_location)), grid_square_size, max_grid_square_inclusive)[0].cpu().numpy()
            for quat_map, pixel_location in zip(quat_maps, pixel_locations)
        ]
        quats = [
            Q.compose_quaternions(
                Q.invert_quaternion(Q.rotation_matrix_to_quaternion(extrinsic[:3, :3])),
                quat/np.linalg.norm(quat, axis=-1, keepdims=True)
            )
            for extrinsic, quat in zip(extrinsics, quats_unrotated)
        ]
        quat = np.mean(quats, axis=0)

        history_quats.append(quat)
        history_2d.append(pixel_locations)

        score = torch.mean(torch.stack(scores), dim=0)

        # Take a step in the direction of the score
        step_size = step_sizes[step]
        pos += step_size * score

        # project to scene bounds
        pos = torch.max(pos, T([xmin, ymin, zmin]))
        pos = torch.min(pos, T([xmax, ymax, zmax]))

        history_3d.append(pos)

    return history_3d, history_quats, history_2d, score_maps

def evaluate(demo, output_prefix):
    device = torch.device("cuda")
    clip = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    model = VisualPlanDiffuserV7(clip).to(device) # type: ignore
    model.load_state_dict(torch.load("diffusion_model.pt"))

    state_action_tuples = create_labels_v3(demo)

    # Test with first keypoint.
    images, positions, quaternions, info = state_action_tuples[0]
    extrinsics = info['extrinsics']
    intrinsics = info['intrinsics']

    starting_point = torch.tensor([0.2, 0, 0.8], device=device)

    images = [prepare_image(image) for image in images]
    positions = torch.tensor(positions) * 224/128
    history, history_quats, history_2d, score_maps = sample(model, images, extrinsics, intrinsics, starting_point)

    # Plot the history
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    viewbox_size = 1 # 112 + some buffer
    ax.set_xlim(-viewbox_size, viewbox_size)
    ax.set_ylim(-viewbox_size, viewbox_size)
    ax.set_zlim(-viewbox_size, viewbox_size)
    ax.plot([point[0].item() for point in history], [point[1].item() for point in history], [point[2].item() for point in history])
    plt.savefig(output_prefix + "_3d_sampling_trajectory.png")
    # plt.show()
    
    grid_size = 14

    plt.clf()
    plt.figure(figsize=(12, 4))

    images = [image.detach().cpu().numpy().transpose(1, 2, 0) for image in images]

    # For debugging the score function.
    for i, (name, history, pos, image, score) in enumerate(zip(
        CAMERAS, zip(*history_2d), positions, images, score_maps
    )):
        position = torch.tensor(pos, device=device)

        scaled_target_position = image_coordinates_to_noise_coordinates(position, image_scaling, center)
        pixel_scaled_positions = torch.zeros((grid_size, grid_size, 2), device=device)
        pixel_scaled_positions[..., 0] = (0.5 + torch.arange(grid_size, device=device).view(grid_size, 1).expand(grid_size, grid_size)) * 2 / grid_size - 1
        pixel_scaled_positions[..., 1] = (0.5 + torch.arange(grid_size, device=device).view(1, grid_size).expand(grid_size, grid_size)) * 2 / grid_size - 1
        pixel_scaled_positions = pixel_scaled_positions.unsqueeze(0).expand(position.shape[0], -1, -1, -1)

        true_direction = (scaled_target_position.view(-1, 1, 1, 2) - pixel_scaled_positions)

        plt.subplot(1, len(images), i + 1)
        plt.title(name + " View")
        plt.imshow(image)
        # plt.imshow(image.detach().cpu().numpy().permute(1, 2, 0), origin="lower")
        plt.scatter(
            [point[0].item() for point in history],
            [point[1].item() for point in history],
            label="Sampling trajectory",
            c=np.arange(len(history)),
            cmap="viridis",
        )
        # Project from 2D memory layout to serial layout
        scaled_arrows(
            image,
            pixel_scaled_positions[0].view(-1, 2),
            score.view(-1, 2),
            true_direction[0].view(-1, 2),
        )
        # Visualize the quaternion
        for j in range(1, len(history_quats)):
            quat_as_matrix = extrinsics[i][:3, :3].T @ Q.quaternion_to_rotation_matrix(history_quats[j])
            
            arrow_scale = 10
            pos = history[j - 1]

            # plot x, y, z axes of this matrix
            # only plot the label if it's the first one to avoid overcrowding the legend
            rotation_matrix_x, rotation_matrix_y, rotation_matrix_z = quat_as_matrix
            plt.quiver(pos[0], pos[1], rotation_matrix_x[0], rotation_matrix_x[1], scale=arrow_scale, color='r', label='x' if j == 1 else None, alpha=0.5)
            plt.quiver(pos[0], pos[1], rotation_matrix_y[0], rotation_matrix_y[1], scale=arrow_scale, color='g', label='y' if j == 1 else None, alpha=0.5)
            plt.quiver(pos[0], pos[1], rotation_matrix_z[0], rotation_matrix_z[1], scale=arrow_scale, color='b', label='z' if j == 1 else None, alpha=0.5)
        plt.legend()


    plt.tight_layout()
    plt.savefig(output_prefix + "_2d_multiview_sampling_trajectory.png", dpi=512)

demos = get_demos()
# print("Running training. [20 epochs]")
# train(demos[:7], epochs=20)
print("Running evaluation on held-out demo.")
evaluate(demos[-1], 'demo_7')
