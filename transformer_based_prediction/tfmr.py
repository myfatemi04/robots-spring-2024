import torch
import torch.nn as nn

class SpatioTemporalTokenizer(nn.Module):
    def __init__(self, token_size: int, n_latent_dims: int, n_image_dims: int):
        super().__init__()

        self.token_size = token_size
        self.n_latent_dims = n_latent_dims
        
        self.convolve = nn.Conv3d(n_image_dims, n_latent_dims, kernel_size=token_size, stride=token_size)

    def forward(self, x):
        return self.convolve(x)

class SpatioTemporalDetokenizer(nn.Module):
    def __init__(self, token_size: int, n_latent_dims: int, n_image_dims: int):
        super().__init__()

        self.token_size = token_size
        self.n_latent_dims = n_latent_dims

        self.deconvolve = nn.ConvTranspose3d(n_latent_dims, n_image_dims, kernel_size=token_size, stride=token_size)

    def forward(self, x):
        return self.deconvolve(x)

class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, n_latent_dims: int, max_width: int, max_height: int, max_temporal_tokens: int):
        self.n_latent_dims = n_latent_dims

        # These will be taken in row-major order
        self.max_spatial_tokens = max_width * max_height
        self.spatial_embedding = nn.Embedding(self.max_spatial_tokens, n_latent_dims)

        self.max_temporal_tokens = max_temporal_tokens
        self.temporal_embedding = nn.Embedding(max_temporal_tokens, n_latent_dims)

    def forward(self, x):
        # x: [batch_size, seqlen, height, width, n_latent_dims]
        (batch_size, seqlen, height, width, n_latent_dims) = x.shape
        
        # Create spatiotemporal additive embeddings
        height_token_ids = torch.arange(0, height).view(height, 1).expand(height, width)
        width_token_ids = torch.arange(0, width).view(1, width).expand(height, width)
        # Each 2D position gets an associated spatial token ID
        spatial_token_ids = height_token_ids * width + width_token_ids
        spatial_tokens = self.spatial_embedding(spatial_token_ids) \
                            .view(1, 1, height, width, n_latent_dims) \
                            .expand(batch_size, seqlen, height, width, n_latent_dims)

        # Create temporal additive embeddings
        temporal_token_ids = torch.arange(0, seqlen)
        temporal_tokens = self.temporal_embedding(temporal_token_ids) \
                            .view(1, seqlen, 1, 1, n_latent_dims) \
                            .expand(batch_size, seqlen, height, width, n_latent_dims)

        x = x + spatial_tokens + temporal_tokens

        return x

class SpatioTemporalAttentionBlock(nn.Module):
    def __init__(self):
        pass

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, token_size: int, n_latent_dims: int, n_image_dims: int):
        super().__init__()

        self.tokenizer = SpatioTemporalTokenizer(token_size, n_latent_dims, n_image_dims)
        self.detokenizer = SpatioTemporalDetokenizer(token_size, n_latent_dims, n_image_dims)

    def forward(self, x):
        pass