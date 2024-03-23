import torch

def calculate_token_coordinates(image_size=224, grid_size=14, device='cuda'):
    grid_square_size = image_size // grid_size
    grid_centers = (torch.arange(grid_size, device=device) + 0.5) * grid_square_size
    token_coordinates = torch.zeros((grid_size, grid_size, 2), device=device)
    token_coordinates[..., 0] = grid_centers.view(grid_size, 1).expand(grid_size, grid_size)
    token_coordinates[..., 1] = grid_centers.view(1, grid_size).expand(grid_size, grid_size)
    
    return token_coordinates


