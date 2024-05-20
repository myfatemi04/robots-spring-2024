import matplotlib.pyplot as plt
import numpy as np
import torch
from detect_objects import clip_processor, clip_text_model, clip_vision_model
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer, CLIPVisionModel
from PIL import Image

import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Adapted from github.com/f3rm/f3rm
def interpolate_positional_embedding(
    positional_embedding: torch.Tensor, x: torch.Tensor, patch_size: int, w: int, h: int
):
    """
    Interpolate the positional encoding for CLIP to the number of patches in the image given width and height.
    Modified from DINO ViT `interpolate_pos_encoding` method.
    https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L174
    """
    assert positional_embedding.ndim == 2, "pos_encoding must be 2D"

    # Number of patches in input
    num_patches = x.shape[1] - 1
    # Original number of patches for square images
    num_og_patches = positional_embedding.shape[0] - 1

    if num_patches == num_og_patches and w == h:
        # No interpolation needed
        return positional_embedding.to(x.dtype)

    dim = x.shape[-1]
    class_pos_embed = positional_embedding[:1]  # (1, dim)
    patch_pos_embed = positional_embedding[1:]  # (num_og_patches, dim)

    # Compute number of tokens
    w0 = w // patch_size
    h0 = h // patch_size
    assert w0 * h0 == num_patches, "Number of patches does not match"

    # Add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1

    # Interpolate
    patch_per_ax = int(np.sqrt(num_og_patches))
    patch_pos_embed_interp = torch.nn.functional.interpolate(
        patch_pos_embed.reshape(1, patch_per_ax, patch_per_ax, dim).permute(0, 3, 1, 2),
        # (1, dim, patch_per_ax, patch_per_ax)
        scale_factor=(w0 / patch_per_ax, h0 / patch_per_ax),
        mode="bicubic",
        align_corners=False,
        recompute_scale_factor=False,
    )  # (1, dim, w0, h0)
    assert (
        int(w0) == patch_pos_embed_interp.shape[-2] and int(h0) == patch_pos_embed_interp.shape[-1]
    ), "Interpolation error."

    patch_pos_embed_interp = patch_pos_embed_interp.permute(0, 2, 3, 1).reshape(-1, dim)  # (w0 * h0, dim)
    # Concat class token embedding and interpolated patch embeddings
    pos_embed_interp = torch.cat([class_pos_embed, patch_pos_embed_interp], dim=0)  # (w0 * h0 + 1, dim)
    return pos_embed_interp.to(x.dtype)

def get_clip_embeddings_dense_hops(image: Image.Image):
    # Encode the image using a hop length of 1/2 the CLIP input size.
    left = 0
    top = 0
    hop = 112
    width, height = image.size
    tiles_width = width // 14
    tiles_height = height // 14
    embedding_count = np.zeros((tiles_height, tiles_width), dtype=int)
    dim = 768 # 1024
    embedding_map = np.zeros((tiles_height, tiles_width, dim))
    while top + 224 <= height:
        while left + 224 <= width:
            print(f"Running CLIP vision model on subimage... {left=}, {top=}")
            sub_image = image.crop((left, top, left + 224, top + 224))
            result = clip_vision_model(
                **clip_processor(images=sub_image, return_tensors='pt'), output_hidden_states=True,
            )
            embedding_map_ = result.hidden_states[-2][0, 1:, :].view(16, 16, -1)
            # apply value projection, following MaskCLIP parameterization trick.
            last_enc: CLIPEncoderLayer = clip_vision_model.vision_model.encoder.layers[-1] # type: ignore
            last_attn: CLIPAttention = clip_vision_model.vision_model.encoder.layers[-1].self_attn
            embedding_map_ = last_enc.layer_norm1(embedding_map_)
            embedding_map_ = last_attn.v_proj(embedding_map_)
            embedding_map_ = last_attn.out_proj(embedding_map_)
            embedding_map_ = clip_vision_model.vision_model.post_layernorm(embedding_map_)
            embedding_map_ = clip_vision_model.visual_projection(embedding_map_)

            # embedding_map_ = result.last_hidden_state[0, 1:, :].view(16, 16, -1)
            
            embedding_map[top // 14:top // 14 + 16, left // 14:left // 14 + 16] += embedding_map_.detach().cpu().numpy()
            embedding_count[top // 14:top // 14 + 16, left // 14:left // 14 + 16] += 1

            left += hop
        left = 0
        top += hop

    embedding_map /= embedding_count[..., None]
    return embedding_map

def embed_interpolated(clip_vision_model: CLIPVisionModel, x: torch.Tensor):
    # Get pre-positional-embedding tokens.
    _, _, h, w = x.shape
    # (b, d, h, w)
    x = clip_vision_model.vision_model.embeddings.patch_embedding(x)
    # (b, d, hw)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    # (b, hw, d)
    x = x.permute(0, 2, 1)
    # add class embedding
    cls_repeated = clip_vision_model.vision_model.embeddings.class_embedding.reshape(1, 1, -1).expand(x.shape[0], -1, -1)
    # prepend image tokens with class embedding
    x = torch.cat([cls_repeated, x], dim=1)

    # Interpolate positional embeddings.
    patch_size = 14
    posenc_interp = interpolate_positional_embedding(
        clip_vision_model.vision_model.embeddings.position_embedding.weight.data,
        x,
        patch_size,
        w,
        h,
    )
    x = x + posenc_interp

    return x

def get_clip_embeddings_dense_interpolated(image: Image.Image):
    x = clip_processor(images=image, return_tensors='pt', output_hidden_states=True, do_center_crop=False).pixel_values.to(device)
    x = embed_interpolated(clip_vision_model, x)
    x = clip_vision_model.vision_model.pre_layrnorm(x)
    result = clip_vision_model.vision_model.encoder(x, output_hidden_states=True)

    h0 = image.height // 14
    w0 = image.width // 14

    embedding_map_ = result.hidden_states[-2][0, 1:, :].view(h0, w0, -1)
    # apply value projection, following MaskCLIP parameterization trick.
    last_enc: CLIPEncoderLayer = clip_vision_model.vision_model.encoder.layers[-1] # type: ignore
    last_attn: CLIPAttention = clip_vision_model.vision_model.encoder.layers[-1].self_attn
    embedding_map_ = last_enc.layer_norm1(embedding_map_)
    embedding_map_ = last_attn.v_proj(embedding_map_)
    embedding_map_ = last_attn.out_proj(embedding_map_)
    embedding_map_ = clip_vision_model.vision_model.post_layernorm(embedding_map_)
    embedding_map_ = clip_vision_model.visual_projection(embedding_map_)
    return embedding_map_.detach().cpu().numpy()

def test():
    image = Image.open("sample_images/oculus_and_headphones.png")
    # This crops to a region of interest near the robot.
    image = image.crop((80 + 112, 720 - 672 + 224, 1200 - 112, 720))
    image = image.resize((224 * 2, 224))

    # plt.title("Image")
    # plt.imshow(image)
    # plt.show()

    text_embedding = clip_text_model(**clip_processor(text="a photo of headphones", return_tensors='pt').to(device)).text_embeds[0, :].detach().cpu().numpy()
    # text_embedding = clip_text_model(**clip_processor(text="a photo of headphones", return_tensors='pt')).text_embeds[0, :].detach().cpu().numpy()

    dense_embeddings = get_clip_embeddings_dense_interpolated(image)
    torch.save(dense_embeddings, "dense_embed.pt")
    # dense_embeddings = torch.load("dense_embed.pt")

    highlight = np.einsum("ijk,k->ij", dense_embeddings, text_embedding)
    highlight -= highlight.min()
    highlight /= highlight.max()
    highlight_img = Image.fromarray((highlight * 255).astype(np.uint8))
    highlight_img = highlight_img.resize((224 * 2, 224))
    highlight = np.array(highlight_img) / 255

    # Highlight embeddings corresponding to controller.
    plt.title("Dense Embeddings")
    plt.imshow(image)
    plt.imshow(highlight, cmap='hot', alpha=highlight)
    plt.show()

if __name__ == '__main__':
    test()
