"""
Experiment 1: Rendering Virtual Plans
"""

"""
Experiment 2: Pretrained Video Prediction Models
 - Video prediction for motion planning: Requires some kind
   of way to control between two predicted frames.
 - We also want to emphasize that by finetuning foundation models,
   we can obtain impressive zero-shot performance.
"""

"""
Experiment 3: Video Infilling as Hierarchical Planning
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.clip.modeling_clip import BaseModelOutput as CLIPBaseModelOutput

class NoiseConditionedScoreModel(nn.Module):
  def __init__(self,
               clip: transformers.CLIPVisionModel,
               clip_processor: transformers.CLIPProcessor,
               diffusion_betas: torch.Tensor,
               max_trajectory_length: int = 32,
               d_model: int = 64,
               num_backbone_heads: int = 4,
               num_backbone_layers: int = 4):
    super().__init__()

    self.clip = clip
    self.clip_processor = clip_processor
    self.d_model = d_model
    self.diffusion_betas = diffusion_betas
    self.diffusion_alpha_cumprods = torch.cumprod(1 - diffusion_betas, dim=0)
    self.temporal_embeddings = nn.Embedding(max_trajectory_length, d_model)

    TRAJECTORY_DIMS = 6
    self.embed_trajectory = nn.Sequential(
      nn.Linear(TRAJECTORY_DIMS, d_model),
      nn.ReLU(),
      nn.Linear(d_model, d_model)
    )
    self.unembed_trajectory_noise = nn.Linear(d_model, TRAJECTORY_DIMS)

    self.downsample_clip_tokens = nn.Linear(512, d_model) if self.d_model != 512 else nn.Identity()

    num_diffusion_steps = len(diffusion_betas)
    self.noise_conditioning_embeddings = nn.Embedding(num_diffusion_steps, d_model)

    self.score_function_backbone = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        d_model,
        nhead=num_backbone_heads,
        dim_feedforward=self.d_model * 4,
      ),
      num_layers=num_backbone_layers,
    )
  
  def predict_noise(self,
                    cue: torch.Tensor,
                    trajectory: torch.Tensor,
                    diffusion_step: torch.Tensor) -> torch.Tensor:
    """
    `cue` is an image.
    `trajectory` is a sequence of [x, y, euler angle, gripper state].

    Input size:
     - `cue`: [batch, image]
     - `trajectory`: [batch, timesteps, TRAJECTORY_DIMS]
    """
    # 1.) Encode the cue image.
    cue_clip_input: BatchEncoding = self.clip_processor(images=[cue], return_tensors="pt")
    cue_clip_output: CLIPBaseModelOutput = self.clip(**cue_clip_input.data)
    cue_clip_tokens_direct = cue_clip_output.last_hidden_state
    num_cue_images, num_clip_tokens, clip_token_dimension = cue_clip_tokens_direct.shape
    cue_clip_tokens = self.downsample_clip_tokens(cue_clip_tokens_direct)

    num_trajectories, trajectory_length, _trajectory_dims = trajectory.shape

    if cue.dim() == 3:
      # No batch dimension for clip.
      # Must repeat the cue tokens.
      # This is useful for training: We can denoise several
      # trajectories for the same input image at once.
      cue_clip_tokens = cue_clip_tokens.unsqueeze(0).repeat((num_trajectories, 1, 1))
    else:
      assert num_cue_images == num_trajectories, "Number of cue images must match number of trajectories, or there must be exactly one cue image provided."

    # 2.) Encode the trajectory with a 2-layer MLP and temporal embeddings.
    trajectory_embedding = self.embed_trajectory(trajectory)
    trajectory_embedding += self.temporal_embeddings(torch.arange(trajectory_length))

    # 3.) Encode the noise conditioning level.
    noise_conditioning_embedding = self.noise_conditioning_embeddings(diffusion_step)

    # 4.) Calculate score function.
    input_tokens = torch.cat([
      cue_clip_tokens,
      trajectory_embedding,
      noise_conditioning_embedding
    ], dim=-2)
    embedded_tokens = self.score_function_backbone(input_tokens)

    # 5.) Unembed trajectory noise.
    trajectory_token_embeddings = embedded_tokens[:, num_clip_tokens:num_clip_tokens + trajectory_length]
    trajectory_noise = self.unembed_trajectory_noise(trajectory_token_embeddings)

    return trajectory_noise

  def loss(self, cue: torch.Tensor, trajectory: torch.Tensor):
    """
    Assumes the trajectory is normalized to [-1, 1].
    """
    
    # Here, we calculate noisy trajectories.
    # We will do this in a vectorized fashion.
    num_diffusion_steps = len(self.diffusion_betas)
    targets = trajectory.unsqueeze(0).repeat((num_diffusion_steps,) + ((1,) * trajectory.dim()))
    noised_trajectories = targets.clone()

    # Now, we add the noise. This is done with the reparametrization trick
    # that allows us to sample at arbitrary diffusion steps.
    # 1. We scale each value by sqrt{^alpha_t}.
    noised_trajectories *= torch.sqrt(self.diffusion_alpha_cumprods)[(None,) * trajectory.dim()]
    # 2. We add noise, with a variance of $1 - \overbar{\alpha}_t$.
    noise = torch.normal(0, 1, size=noised_trajectories.shape)
    noise *= torch.sqrt(1 - self.diffusion_alpha_cumprods)[(None,) * trajectory.dim()]
    noised_trajectories += noise

    # Now, we calculate the loss: MSE over all timesteps.
    # We need a separate loss value p(x0 | x1), as there is no variance added.
    # Perhaps we can just leave it as MSE too... and just not add variance
    # during inference?
    predicted_noise = self.predict_noise(cue, noised_trajectories, torch.arange(num_diffusion_steps))
    # Take mean across sequence step and trajectory dims.
    # At some point, we may wish to have different loss functions for different
    # dimensions of the trajectory and/or time steps. But for now, it's just
    # a mean.
    error_per_diffusion_step = ((predicted_noise - noise) ** 2).mean((-2, -1))
    # Using Jonathan Ho's simplified loss, we can actually avoid weighing by which diffusion step we're at...
    # We can just take the mean across all diffusion steps.
    loss = error_per_diffusion_step.mean()

    return loss

# Starting with a 3D trajectory and camera setup, we gradually add noise to the points.

input_folder = "control_labels/trajectories/003_grab_water_bottle_by_side"
with open(input_folder + "/data.json", "r") as f:
  trajectory = json.load(f)

# dict_keys(['times', 'poses', 'quaternions', 'gripper_widths', 'gripper_fully_closed_width', 'gripper_fully_open_width'])
# print(trajectory.keys())

