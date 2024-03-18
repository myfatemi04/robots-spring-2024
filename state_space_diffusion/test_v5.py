### Goal: Make something that can effectively denoise towards the point (2/3, 2/3).

from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import tqdm
import transformers
from get_data import get_data
from model_architectures import VisualPlanDiffuserV5


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

clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16") # type: ignore

for name, param in model.named_parameters():
    # check if MLP
    if "linear" in name and ('.weight' in name or '.bias' in name):
        param.data.fill_(0)

# model = torch.nn.Parameter(torch.zeros((14, 14, 2), device=device))
# optim = torch.optim.Adam([model], lr=1e-3)

optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
# optim = NoamOpt(768, 1, 200, torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0, betas=(0.9, 0.98), eps=1e-9))

dataset = get_data()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

image_scaling = 112
center = 112
grid_size = 14

betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_diffusion_schedule()

NI = partial(noise_coordinates_to_image_coordinates, image_scaling=image_scaling, center=center)
NP = lambda x: x.detach().cpu().numpy()

def scaled_arrows(pixel_scaled_positions, pred_direction, true_direction):
    plt.imshow(pixel_values[0].detach().cpu().numpy().transpose(1, 2, 0), origin="lower")

    pred_arrow_ends = NI(pixel_scaled_positions)
    true_arrow_ends = NI(pixel_scaled_positions)

    plt.quiver(
        NP(pred_arrow_ends[:, 0]),
        NP(pred_arrow_ends[:, 1]),
        -NP(pred_direction[:, 0]) * image_scaling,
        -NP(pred_direction[:, 1]) * image_scaling,
        color='red',
        label='Predicted'
    )
    plt.quiver(
        NP(true_arrow_ends[:, 0]),
        NP(true_arrow_ends[:, 1]),
        -NP(true_direction[:, 0]) * image_scaling,
        -NP(true_direction[:, 1]) * image_scaling,
        color='blue',
        label='True'
    )

for epoch in range(100):
    for (image, position) in (pbar := tqdm.tqdm(dataloader)):
        pixel_values = clip_processor(images=image, return_tensors="pt", do_rescale=False).to(device=device).pixel_values

        scaled_position = image_coordinates_to_noise_coordinates(position, image_scaling, center)
        pixel_scaled_positions = torch.zeros((grid_size, grid_size, 2), device=device)
        pixel_scaled_positions[..., 0] = (0.5 + torch.arange(grid_size, device=device).view(1, grid_size).expand(grid_size, grid_size)) * 2 / grid_size - 1
        pixel_scaled_positions[..., 1] = (0.5 + torch.arange(grid_size, device=device).view(grid_size, 1).expand(grid_size, grid_size)) * 2 / grid_size - 1
        pixel_scaled_positions = pixel_scaled_positions.unsqueeze(0).expand(position.shape[0], -1, -1, -1)

        true_direction = (pixel_scaled_positions - scaled_position.view(-1, 1, 1, 2))

        # pred_direction = model.view(-1, grid_size, grid_size, 2)
        pred_direction = model(pixel_values).view(-1, grid_size, grid_size, 2)

        loss = (pred_direction - true_direction).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        pbar.set_postfix({"loss": loss.item()})

    # Visualize the vector field
    if (epoch + 1) % 10 == 0:
        plt.clf()
        plt.title("Score Function")
        # Project from 2D memory layout to serial layout
        scaled_arrows(
            pixel_scaled_positions[0].view(-1, 2),
            pred_direction[0].view(-1, 2),
            true_direction[0].view(-1, 2),
        )
        plt.legend()
        plt.savefig(f"diffusion_{epoch}.png")
