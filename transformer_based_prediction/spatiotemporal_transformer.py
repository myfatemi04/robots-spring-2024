import torch
import torch.nn as nn

class SpatioTemporalTokenizer(nn.Module):
    def __init__(self, token_size: int, n_latent_dims: int, n_image_dims: int):
        super().__init__()

        self.token_size = token_size
        self.n_latent_dims = n_latent_dims
        
        self.convolve = nn.Conv3d(n_image_dims, n_latent_dims, kernel_size=token_size, stride=token_size)

    def forward(self, x):
        # (batch_size, seqlen, n_image_dims, height, width) = x.shape
        # Permute to [batch_size, n_image_dims, seqlen, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.convolve(x)
        # Permute back to [batch_size, seqlen, n_latent_dims, height, width]
        x = x.permute(0, 2, 1, 3, 4)

        return x

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
        super().__init__()

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

        # Reshape to [batch_size, seqlen, height * width, n_latent_dims]
        x = x.view(batch_size, seqlen, height * width, self.n_latent_dims)
        x = x.contiguous()

        return x

class SpatioTemporalAttentionBlock(nn.Module):
    def __init__(self, n_latent_dims: int, n_spatial_attention_heads: int, n_temporal_attention_heads: int, n_mlp_dims: int):
        super().__init__()
        
        self.n_latent_dims = n_latent_dims
        self.n_spatial_attention_heads = n_spatial_attention_heads
        self.n_temporal_attention_heads = n_temporal_attention_heads

        self.spatial_attention = nn.MultiheadAttention(n_latent_dims, n_spatial_attention_heads, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(n_latent_dims, n_temporal_attention_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(n_latent_dims, n_mlp_dims),
            nn.ReLU(),
            nn.Linear(n_mlp_dims, n_latent_dims)
        )

    def forward(self, x):
        # x: [batch_size, seqlen, n_frame_tokens, n_latent_dims]
        (batch_size, sequence_length, n_frame_tokens, n_latent_dims) = x.shape

        # Spatial attention
        # Make seqlen a batch dimension by reshaping to [batch_size * seqlen, height * width, n_latent_dims]
        x = x.view(batch_size * sequence_length, n_frame_tokens, n_latent_dims)
        x, _ = self.spatial_attention(x, x, x)
        x = x.view(batch_size, sequence_length, n_frame_tokens, n_latent_dims)

        # Temporal attention
        # Apply causal attention mask
        causal_mask = torch.tril(torch.ones(sequence_length, sequence_length)).to(x.device)
        # Permute to [batch_size, n_frame_tokens, seqlen, n_latent_dims]
        x = x.permute(0, 2, 3, 1, 4)
        # Make height * width a batch dimension by reshaping to [batch_size * height * width, seqlen, n_latent_dims]
        x = x.view(batch_size * n_frame_tokens, sequence_length, n_latent_dims)
        x, _ = self.temporal_attention(x, x, x, attn_mask=causal_mask)
        x = x.view(batch_size, n_frame_tokens, sequence_length, n_latent_dims)
        # Permute back to [batch_size, seqlen, n_frame_tokens, n_latent_dims]
        x = x.permute(0, 3, 1, 2, 4)

        # Linear residual layer
        x = x + self.mlp(x)

        return x

class SpatioTemporalTransformer(nn.Module):
    def __init__(self,
                 token_size: int,
                 n_latent_dims: int,
                 n_image_dims: int,
                 max_width: int,
                 max_height: int,
                 max_temporal_tokens: int,
                 n_spatial_attention_heads: int,
                 n_temporal_attention_heads: int,
                 n_mlp_dims: int,
                 n_layers: int,
                ):
        super().__init__()

        self.tokenizer = SpatioTemporalTokenizer(token_size, n_latent_dims, n_image_dims)
        self.embedder = SpatioTemporalEmbedding(n_latent_dims, max_width, max_height, max_temporal_tokens)
        self.detokenizer = SpatioTemporalDetokenizer(token_size, n_latent_dims, n_image_dims)

        self.attention_blocks = nn.ModuleList([
            SpatioTemporalAttentionBlock(
                n_latent_dims,
                n_spatial_attention_heads,
                n_temporal_attention_heads,
                n_mlp_dims
            )
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.embedder(x)
        for attention_block in self.attention_blocks:
            x = attention_block(x)
        x = self.detokenizer(x)
        return x
