import torch

class Patchify(torch.nn.Module):
    def __init__(self, token_size: int, image_size_tokens: int, dim: int = 256, input_dim: int = 3):
        super().__init__()

        self.token_size = token_size
        self.image_size_tokens = image_size_tokens
        self.dim = dim
        self.conv = torch.nn.Conv2d(input_dim, dim, kernel_size=token_size, stride=token_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [batch, input_dim, image_size_tokens * token_size, image_size_tokens * token_size]
        # output: [batch, image_size_tokens, image_size_tokens, dim]
        output = self.conv(images)
        # move channel dimension last
        output = output.permute(0, 2, 3, 1)
        # flatten middle two dimensions
        output = output.reshape(images.shape[0], self.image_size_tokens * self.image_size_tokens, self.dim)
        return output

class MVT(torch.nn.Module):
    def __init__(self, dim: int = 256, num_views: int = 3, token_size: int = 32, image_size_tokens: int = 7):
        super().__init__()

        self.dim = dim
        self.num_views = num_views
        self.patchify = Patchify(token_size, image_size_tokens, dim, 3)
        self.positional_embedding = torch.nn.Embedding(256 * 3, dim)
        # We will use this transformer encoder layer with a custom mask
        # for the different types of attentions used
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(dim, nhead=8, dim_feedforward=dim * 4, batch_first=True)
            for _ in range(8)
        ])
        self.projection = torch.nn.Linear(dim, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [batch, num_views, 3, 256, 256]
        # output: [batch, num_views, 16 * 16, dim]
        image_size = self.patchify.image_size_tokens * self.patchify.token_size
        images_batch_flattened = images.reshape(-1, 3, image_size, image_size)
        patches_batch_flattened = self.patchify(images_batch_flattened)
        patches = patches_batch_flattened.reshape(images.shape[0], self.num_views, -1, self.dim)
        tokens_per_image = self.patchify.image_size_tokens * self.patchify.image_size_tokens
        # flatten view and token dimensions
        patches = patches.reshape(
            # batch
            images.shape[0],
            # num_views, h, w -> (num_views * h * w)
            self.num_views * tokens_per_image,
            self.dim
        )
        # add positional embeddings
        patches += self.positional_embedding(torch.arange(patches.shape[1], device=patches.device))
        # create mask for first four layers
        intra_image_mask = torch.zeros(patches.shape[1], patches.shape[1], device=patches.device)
        for i in range(3):
            intra_image_mask[
                i * tokens_per_image:(i + 1) * tokens_per_image,
                i * tokens_per_image:(i + 1) * tokens_per_image
            ] = 1
        # we will use no mask for the last four layers, as all tokens can communicate with each other
        # run transformer encoder layers
        for i in range(8):
            layer = self.layers[i]
            # assertion to allow runtime-safe type inference
            assert isinstance(layer, torch.nn.TransformerEncoderLayer)
            patches = layer(patches, src_mask=intra_image_mask if i < 4 else None)
        # project to heatmap
        patches = self.projection(patches)
        # reconstruct original image structure
        patches = patches.reshape(
            images.shape[0], # batch
            self.num_views,
            self.patchify.image_size_tokens,
            self.patchify.image_size_tokens,
            1 # encoded token dim (1 for the heatmap)
        )
        return patches
