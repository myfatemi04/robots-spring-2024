import torch
import torch.nn as nn
import transformers
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPVisionConfig, CLIPVisionModel
from positional_encoding import PositionalEncoding

# Naïve approach that will surely be updated.
class VisualPlanDiffuserV1(torch.nn.Module):
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
class VisualPlanDiffuserV2Categorical(torch.nn.Module):
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

def render_noisy_state(position: torch.Tensor, sigma: torch.Tensor):
    """
    Renders as a Gaussian.
    """
    device = position.device

    # position: [batch, 2] -> [batch, 2, 1, 1]
    position = position.view(-1, 2, 1, 1)

    pixel_positions = torch.zeros((position.shape[0], 2, 224, 224), device=device)
    # [224] -> [1, 1, 1, 224] -> [batch, 1, 224, 224]
    pixel_positions[:, 0, :, :] = torch.arange(0, 224, device=device).view(1, 224)
    # [224] -> [1, 1, 224, 1] -> [batch, 1, 224, 224]
    pixel_positions[:, 1, :, :] = torch.arange(0, 224, device=device).view(224, 1)

    # compute probability density function.
    z_map = (pixel_positions - position).norm(dim=1, keepdim=True) / sigma.view(-1, 1, 1, 1)
    pdf = torch.exp(-0.5 * z_map.pow(2)) / (2 * torch.pi * sigma.view(-1, 1, 1, 1))

    return pdf

# continuous but with a different positional encoding
class VisualPlanDiffuserV3(torch.nn.Module):
    def __init__(self, clip: transformers.CLIPVisionModel):
        super().__init__()

        self.clip = clip
        d_model = clip.config.hidden_size

        # Noisy state gets input as an image.
        new_config = CLIPVisionConfig(**clip.vision_model.config.to_dict())
        new_config.num_channels = 1
        self.noisy_state_embeddings = CLIPVisionEmbeddings(new_config)

        # Denoising.
        self.timestep_encoding = PositionalEncoding(n_position_dims=1, n_encoding_dims=d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=8, dim_feedforward=48, batch_first=True),
            num_layers=6,
        )
        self.noise_decoder = nn.Linear(d_model, 2)

    def forward(self, pixel_values: torch.FloatTensor, noisy_state: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Encode the image, creating image tokens.
        with torch.no_grad():
            image_features = self.clip.forward(pixel_values).last_hidden_state # type: ignore

        noisy_state_tokens = self.noisy_state_embeddings(render_noisy_state(noisy_state, sigma))
        noisy_state_tokens_encoded = self.transformer_decoder.forward(noisy_state_tokens, image_features)
        noisy_state_tokens_noise_pred = self.noise_decoder(noisy_state_tokens_encoded[:, 0, :])

        return noisy_state_tokens_noise_pred

# continuous and using a single transformer for the whole fwd pass (simulation just looks bad)
class VisualPlanDiffuserV4(torch.nn.Module):
    def __init__(self, d_model, vision_config: CLIPVisionConfig):
        super().__init__()

        # Noisy state gets input as an image.
        new_config = CLIPVisionConfig(**vision_config.to_dict())
        new_config.num_channels = 4
        self.noisy_state_embeddings = CLIPVisionEmbeddings(new_config)

        # Denoising.
        self.timestep_encoding = PositionalEncoding(n_position_dims=1, n_encoding_dims=d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=4,
        )
        self.noise_decoder = nn.Linear(d_model, 2)

    def forward(self, pixel_values: torch.FloatTensor, noisy_state: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Encode the image, creating image tokens.
        # with torch.no_grad():
        #     image_features = self.clip.forward(pixel_values).last_hidden_state # type: ignore
        combination_state = torch.cat([pixel_values, render_noisy_state(noisy_state, sigma)], dim=1)

        noisy_state_tokens = self.noisy_state_embeddings(combination_state)
        noisy_state_tokens_encoded = self.transformer_encoder.forward(noisy_state_tokens)
        noisy_state_tokens_noise_pred = self.noise_decoder(noisy_state_tokens_encoded[:, 0, :])

        return noisy_state_tokens_noise_pred

# continuous and predicting direction of nearest planning mode
class VisualPlanDiffuserV5(torch.nn.Module):
    def __init__(self, d_model, vision_config: CLIPVisionConfig):
        super().__init__()

        # Noisy state gets input as an image.
        self.noisy_state_embeddings = CLIPVisionEmbeddings(vision_config)

        # Denoising.
        self.timestep_encoding = PositionalEncoding(n_position_dims=1, n_encoding_dims=d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=4,
        )
        self.noise_decoder = nn.Linear(d_model, 2)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        noisy_state_tokens = self.noisy_state_embeddings(pixel_values)
        noisy_state_tokens_encoded = self.transformer_encoder(noisy_state_tokens)
        noisy_state_tokens_gradient_pred = self.noise_decoder(noisy_state_tokens_encoded[:, 1:, :])

        return noisy_state_tokens_gradient_pred

# continuous and predicting direction of nearest planning mode; but using CLIP as a backbone
class VisualPlanDiffuserV6(torch.nn.Module):
    def __init__(self, clip: CLIPVisionModel):
        super().__init__()

        d_model = clip.config.hidden_size
        
        self.tfmr = clip
        self.noise_decoder = nn.Linear(d_model, 2)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        return self.noise_decoder(self.tfmr(pixel_values).last_hidden_state[:, 1:, :])
