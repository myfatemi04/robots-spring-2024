### Goal: Make something that can effectively denoise towards the point (2/3, 2/3).

import torch
import transformers
from model_architectures import VisualPlanDiffuserV4, render_noisy_state


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
model = VisualPlanDiffuserV4(768, cfg).to(device)
optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_diffusion_schedule()

for epoch in range(100000):
    pixel_values = torch.zeros((1, 3, 224, 224), device=device)

    position_noise_space = torch.tensor([1/3, 1/3]).unsqueeze(0).to(device)
    timestep = torch.randint(0, 10, (1,)).to(device)

    # std.dev of noise
    sigma = sqrt_one_minus_alphas_cumprod[timestep].unsqueeze(-1)
    noise = torch.randn_like(position_noise_space.float())

    noisy_position = get_noisy_position(position_noise_space, noise, sigma)

    image_scaling = 112
    center = 112
    noise_pred = model(
        pixel_values,
        noise_coordinates_to_image_coordinates(noisy_position, image_scaling, center),
        sigma
    )

    loss = ((noise_pred - noisy_position).pow(2) / (sigma.view(-1, 1) ** 2 + 1)).mean()

    print(noisy_position, noise_pred)

    optim.zero_grad()
    loss.backward()
    optim.step()

    print(loss.item())
