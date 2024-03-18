import torch
import torch.nn as nn
import transformers
from positional_encoding import PositionalEncoding

# Naïve approach that will surely be updated.
class VisualPlanDiffuser(torch.nn.Module):
    def __init__(self, clip: transformers.CLIPVisionModel):
        super().__init__()

        self.clip = clip
        d_model = clip.config.hidden_size
        # Token used for predicting noise.
        self.timestep_encoding = PositionalEncoding(n_position_dims=1, n_encoding_dims=d_model)
        self.noise_decoder = nn.Linear(d_model, 2)
        self.positional_encoding = PositionalEncoding(n_position_dims=2, n_encoding_dims=d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=2, batch_first=True),
            num_layers=2,
        )

    def forward(self, pixel_values: torch.FloatTensor, noisy_state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # Encode the image, creating image tokens.
        with torch.no_grad():
            image_features = self.clip.forward(pixel_values).last_hidden_state # type: ignore

        # print(image_features.shape)
        
        # Use spatial features to denoise plan location.
        current_plan_token = self.positional_encoding(noisy_state * 50)

        # print(current_plan_token.shape)

        # both are [batch, 1, d_model]
        noise_tok = self.timestep_encoding(timestep).expand(pixel_values.shape[0], -1, -1)
        state_tok = current_plan_token.unsqueeze(1)

        # print(noise_tok.shape, state_tok.shape)

        decoder_inputs = torch.cat([state_tok, noise_tok], dim=1)

        decoded = self.transformer_decoder.forward(decoder_inputs, image_features)
        latent_noise_pred = decoded[:, 1, :]

        # print(decoded.shape, latent_noise_pred.shape)
        
        predicted_noise = self.noise_decoder(latent_noise_pred)

        # print(predicted_noise.shape)
        
        return predicted_noise

# Predicts whether the true direction is up/down and left/right
class VisualPlanDiffuserCategorical(torch.nn.Module):
    def __init__(self, clip: transformers.CLIPVisionModel):
        super().__init__()

        self.clip = clip
        d_model = clip.config.hidden_size
        # Token used for predicting noise.
        self.timestep_encoding = PositionalEncoding(n_position_dims=1, n_encoding_dims=d_model)
        self.noise_decoder = nn.Linear(d_model, 4)
        self.positional_encoding = PositionalEncoding(n_position_dims=2, n_encoding_dims=d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=8, dim_feedforward=48, batch_first=True),
            num_layers=6,
        )

    def forward(self, pixel_values: torch.FloatTensor, noisy_state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # Encode the image, creating image tokens.
        with torch.no_grad():
            image_features = self.clip.forward(pixel_values).last_hidden_state # type: ignore

        # Use spatial features to denoise plan location.
        current_plan_token = self.positional_encoding(noisy_state * 50)

        # both are [batch, 1, d_model]
        noise_tok = self.timestep_encoding(timestep).expand(pixel_values.shape[0], -1, -1)
        state_tok = current_plan_token.unsqueeze(1)

        decoder_inputs = torch.cat([state_tok, noise_tok], dim=1)

        decoded = self.transformer_decoder.forward(decoder_inputs, image_features)
        latent_noise_pred = decoded[:, 1, :]

        predicted_noise = self.noise_decoder(latent_noise_pred)

        return predicted_noise


