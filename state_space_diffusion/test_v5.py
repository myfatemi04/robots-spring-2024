### Goal: Make something that can effectively denoise towards the point (2/3, 2/3).

import torch
import tqdm
import transformers
from model_architectures import VisualPlanDiffuserV5
import matplotlib.pyplot as plt
from noamopt import NoamOpt


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

device = torch.device('cuda')

cfg = transformers.CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch16")
model = VisualPlanDiffuserV5(768, cfg).to(device)

# for name, param in model.named_parameters():
#     # check if MLP
#     if "linear1.weight" in name or "linear2.weight" in name:
#         param.data.fill_(0)
#         param.data.fill_(0)

# model = torch.nn.Parameter(torch.zeros((14, 14, 2), device=device))
# optim = torch.optim.Adam([model], lr=1e-3)

optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
# optim = NoamOpt(768, 1, 200, torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0, betas=(0.9, 0.98), eps=1e-9))

betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_diffusion_schedule()

for epoch in (pbar := tqdm.tqdm(range(1000))):
    pixel_values = torch.zeros((1, 3, 224, 224), device=device)

    position = torch.tensor([1/3, 1/3]).unsqueeze(0).to(device)
    grid_size = 14
    pixel_locations = torch.zeros((grid_size, grid_size, 2), device=device)
    pixel_locations[..., 0] = torch.arange(grid_size, device=device).view(1, grid_size).expand(grid_size, grid_size) / grid_size
    pixel_locations[..., 1] = torch.arange(grid_size, device=device).view(grid_size, 1).expand(grid_size, grid_size) / grid_size
    true_direction = (pixel_locations - position)

    image_scaling = 112
    center = 112
    # pred_direction = model.view(-1, grid_size, grid_size, 2)
    pred_direction = model(pixel_values).view(-1, grid_size, grid_size, 2)

    loss = (pred_direction - true_direction).pow(2).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()

    # Visualize the vector field
    if (epoch + 1) % 100 == 0:
        print(pred_direction.mean(dim=(0, 1, 2)), true_direction.mean(dim=(0, 1)))
        plt.clf()
        plt.title("Score Function")
        plt.quiver(
            pixel_locations[..., 0].cpu().numpy(),
            pixel_locations[..., 1].cpu().numpy(),
            pred_direction[0, ..., 0].detach().cpu().numpy(),
            pred_direction[0, ..., 1].detach().cpu().numpy(),
            color='red',
            label='Predicted'
        )
        plt.quiver(
            pixel_locations[..., 0].cpu().numpy(),
            pixel_locations[..., 1].cpu().numpy(),
            true_direction[..., 0].cpu().numpy(),
            true_direction[..., 1].cpu().numpy(),
            color='blue',
            label='True'
        )
        plt.legend()
        plt.savefig(f"diffusion_{epoch}.png")
    
    pbar.set_postfix({"loss": loss.item()})
