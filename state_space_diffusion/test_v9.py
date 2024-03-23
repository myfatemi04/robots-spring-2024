### v9: multi-task training

import matplotlib.pyplot as plt
import numpy as np
import quaternions as Q
import torch
import torch.utils.data
import tqdm
import transformers
from demo_to_state_action_pairs import create_torch_dataset_v3
from get_data import generate_demos
from model_architectures import VisualPlanDiffuserV7
from visualization import visualize_2d_score_prediction
import pickle
import transformers

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

    # Generates (image, position, quaternion) triplets
    dataset = create_torch_dataset_v3(demos, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    # We predict the direction of the keypoint from *these* pixel locations.
    # These are the 2D coordinates of each token, in pixel space.
    token_coordinates = grid_sampling.calculate_token_coordinates(image_size=224, grid_size=14, device=device)

    for epoch in range(epochs):
        for (image, position, quat) in (pbar := tqdm.tqdm(dataloader)):
            pixel_values = clip_processor(images=image, return_tensors="pt", do_rescale=False).to(device=device).pixel_values # type: ignore
            
            # Unsqueeze for this particular batch
            token_coordinates_ = token_coordinates.unsqueeze(0).expand(position.shape[0], -1, -1, -1)
            position_ = position.view(-1, 1, 1, 2)

            true_direction = (position_ - token_coordinates_)
            true_direction_scaled = true_direction / (image_size / 2)

            # (batch, token_y, token_x, 2) -> (batch, token_x, token_y, 2)
            pred = model(pixel_values).view(-1, grid_size, grid_size, 6).permute(0, 2, 1, 3)
            pred_direction = pred[..., 0:2].contiguous()
            pred_quat = pred[..., 2:6].contiguous()
            
            pred_direction_px = pred_direction * (image_size / 2)

            score_loss = (pred_direction - true_direction_scaled).pow(2).mean()
            
            # Weighted average based on distance to keypoint.
            distances = (token_coordinates_ - position_).norm(dim=-1, keepdim=True)
            quat_loss_weighting = torch.exp(-distances / 0.1)
            quat_loss_weighting = quat_loss_weighting / quat_loss_weighting.sum(dim=(1, 2, 3), keepdim=True)
            quat = quat.view(-1, 1, 1, 4)
            quat_loss = ((pred_quat - quat).pow(2) * quat_loss_weighting).mean()

            quat_loss_coeff = 10
            loss = score_loss + quat_loss_coeff * quat_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            pbar.set_postfix({"loss": loss.item(), "quat_loss": quat_loss.item(), "score_loss": score_loss.item()})

        # Visualize the vector field
        if (epoch + 1) % 10 == 0:
            plt.clf()
            plt.title("Score Function")
            visualize_2d_score_prediction(
                image[0],
                token_coordinates.view(-1, 2),
                pred_direction_px[0].view(-1, 2),
                true_direction[0].view(-1, 2),
            )
            plt.legend()
            plt.savefig(f"diffusion_{epoch}.png")

    torch.save(model.state_dict(), "diffusion_model.pt")

# demos = generate_demos("close_jar", 10)

# with open("replay_buffer/close_jar_10.pkl", "wb") as f:
#     pickle.dump(demos, f)

with open("replay_buffer/close_jar_10.pkl", "rb") as f:
    demos = pickle.load(f)

# train(demos[:8])

print("Loading checkpoints.")
device = torch.device("cuda")
clip = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
model = VisualPlanDiffuserV7(clip).to(device)
model.load_state_dict(torch.load("diffusion_model.pt"))
print("Loaded checkpoints.")

import inference
import visualization
from demo_to_state_action_pairs import create_labels_v3, CAMERAS as camera_names

demo = demos[8]
tuples = create_labels_v3(demo)
(images, target_locations_2d, quaternions, info) = tuples[0]
extrinsics = info['extrinsics']
intrinsics = info['intrinsics']

print("Sampling keypoint.")
(history_3d, history_quaternions, history_2d_projections, predicted_score_maps, quat_maps) = \
    inference.sample_keypoint(model, images, extrinsics, intrinsics, (0.5, 0.5, 1.0))
print("Finished sampling keypoint.")

batch_size = len(images)
token_coordinates = grid_sampling.calculate_token_coordinates(image_size=224, grid_size=14, device=device)
true_score_maps = token_coordinates.view(1, 14, 14, 2).expand(batch_size, -1, -1, -1) - target_locations_2d.view(batch_size, 1, 1, 2)

visualization.visualize_2d_sampling_trajectory(images, predicted_score_maps, true_score_maps, history_2d_projections, camera_names)
