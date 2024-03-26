### v9: multi-task training

import matplotlib.pyplot as plt
import torch
import torch.utils.data
import tqdm
import transformers
from demo_to_state_action_pairs import create_labels_v4, create_torch_dataset_v4, CAMERAS as camera_names
from get_data import generate_demos
from model_architectures import VisualPlanDiffuserV7
from visualization import visualize_2d_score_prediction
import pickle
import transformers
import grid_sampling
import inference
import visualization
import grid_sampling

def freeze_clip_layers(clip, first_n=0):
    for i in range(first_n):
        clip.vision_model.encoder.layers[i].requires_grad_(False)
    print(f"Freezing first {first_n} layers out of {len(clip.vision_model.encoder.layers)}")
    
def annihilate_mlp_parameters(model):
    for name, param in model.named_parameters():
        # check if MLP
        if "linear" in name and ('.weight' in name or '.bias' in name):
            param.data.fill_(0)

def train(demos, epochs=10, model=None, include_original=True, include_virtual=False):
    device = torch.device('cuda')

    clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16") # type: ignore
    if model is None:
        clip = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device) # type: ignore
        model = VisualPlanDiffuserV7(clip).to(device)

    freeze_clip_layers(model.tfmr, first_n=0)
    # annihilate_mlp_parameters(model)

    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    # Generates (image, position, quaternion) triplets
    dataset = create_torch_dataset_v4(demos, device=device, include_original=include_original, include_virtual=include_virtual)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    # We predict the direction of the keypoint from *these* pixel locations.
    # These are the 2D coordinates of each token, in pixel space.
    image_size = 224
    grid_size = 14
    token_coordinates = grid_sampling.calculate_token_coordinates(image_size, grid_size, device=device)

    for epoch in range(epochs):
        for (image, position, quat) in (pbar := tqdm.tqdm(dataloader)):
            pixel_values = clip_processor(images=image, return_tensors="pt", do_rescale=False).to(device=device).pixel_values # type: ignore
            
            # Unsqueeze for this particular batch
            token_coordinates_ = token_coordinates.unsqueeze(0).expand(position.shape[0], -1, -1, -1)
            position_ = position.view(-1, 1, 1, 2)

            true_direction = (position_ - token_coordinates_)
            true_direction_scaled = true_direction / (image_size / 2)
            # prevent losing important information close to the minimum
            true_direction_scaled_2 = true_direction_scaled / true_direction_scaled.norm(dim=-1, keepdim=True).pow(0.5)
            
            distances = (token_coordinates_ - position_).norm(dim=-1, keepdim=True) / (image_size / 2)
            loss_weighting = torch.exp(-distances)
            
            # (batch, token_y, token_x, 2) -> (batch, token_x, token_y, 2)
            pred = model(pixel_values).view(-1, grid_size, grid_size, 6).permute(0, 2, 1, 3)
            pred_direction = pred[..., 0:2].contiguous()
            pred_quat = pred[..., 2:6].contiguous()
            
            pred_direction_px = pred_direction * (image_size / 2)

            score_loss = ((pred_direction - true_direction_scaled_2).pow(2) * loss_weighting).mean()
            
            # Weighted average based on distance to keypoint.
            loss_weighting = loss_weighting / loss_weighting.sum(dim=(1, 2, 3), keepdim=True)
            quat = quat.view(-1, 1, 1, 4)
            quat_loss = ((pred_quat - quat).pow(2) * loss_weighting).mean()
            
            assert not torch.any(quat.isnan()), "quat is nan"
            # assert not torch.any(quat_loss_weighting.isnan()), "quat_loss_weighting is nan"
            assert not torch.any(pred_quat.isnan()), "pred_quat is nan"
            assert not torch.any(distances.isnan()), "distances is nan"
            assert not torch.any(distances < 0), "distance < 0"

            quat_loss_coeff = 10
            loss = score_loss # + quat_loss_coeff * quat_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            pbar.set_postfix({"loss": loss.item(), "quat_loss": quat_loss.item(), "score_loss": score_loss.item()})

        # Visualize the vector field
        if (epoch + 1) % 5 == 0:
            plt.clf()
            plt.title("Score Function")
            visualize_2d_score_prediction(
                image[0],
                token_coordinates.view(-1, 2),
                pred_direction_px[0].view(-1, 2),
                true_direction[0].view(-1, 2),
            )
            plt.legend()
            plt.savefig(f"diffusion_{epoch + 1}.png")

    torch.save(model.state_dict(), "diffusion_model.pt")

if __name__ == '__main__':
    # demos = generate_demos("close_jar", 10)

    # with open("replay_buffer/close_jar_10.pkl", "wb") as f:
    #     pickle.dump(demos, f)

    with open("replay_buffer/close_jar_10.pkl", "rb") as f:
        demos = pickle.load(f)

    # train(demos[:1], epochs=5)

    print("Loading checkpoints.")
    device = torch.device("cuda")
    clip = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    model = VisualPlanDiffuserV7(clip).to(device)
    model.load_state_dict(torch.load("diffusion_model.pt"))
    print("Loaded checkpoints.")

    demo = demos[0]
    original_tuples, virtual_tuples = create_labels_v4(demo)
    (images, target_locations_2d, quaternions, info) = original_tuples[2]
    extrinsics = info['extrinsics']
    intrinsics = info['intrinsics']
    target_locations_2d = torch.tensor(target_locations_2d, device=device)

    print("Sampling keypoint.")
    (history_3d, history_quaternions, history_2d_projections, predicted_score_maps, quat_maps) = \
        inference.sample_keypoint(model, images, extrinsics, intrinsics, (0.5, 0.5, 1.0))
    print("Finished sampling keypoint.")

    batch_size = len(images)
    token_coordinates = grid_sampling.calculate_token_coordinates(image_size=224, grid_size=14, device=device)
    true_score_maps = target_locations_2d.view(batch_size, 1, 1, 2) - token_coordinates.view(1, 14, 14, 2).expand(batch_size, -1, -1, -1)

    plt.figure(figsize=(16, 4))

    visualization.visualize_2d_sampling_trajectory(images, token_coordinates, predicted_score_maps, true_score_maps, history_2d_projections, camera_names)

    plt.tight_layout()
    plt.savefig("0_visualization.png", dpi=64)
