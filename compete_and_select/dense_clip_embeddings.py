import matplotlib.pyplot as plt
import numpy as np
import torch
from detect_objects import clip_processor, clip_text_model, clip_vision_model
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer
from PIL import Image


def get_clip_embeddings_dense(image: Image.Image):
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

def test():
    image = Image.open("sample_images/oculus_and_headphones.png")
    # This crops to a region of interest near the robot.
    image = image.crop((80 + 112, 720 - 672 + 224, 1200 - 112, 720))
    image = image.resize((224 * 2, 224))

    # plt.title("Image")
    # plt.imshow(image)
    # plt.show()

    text_embedding = clip_text_model(**clip_processor(text="a photo of a drawer", return_tensors='pt')).text_embeds[0, :].detach().cpu().numpy()
    # text_embedding = clip_text_model(**clip_processor(text="a photo of headphones", return_tensors='pt')).text_embeds[0, :].detach().cpu().numpy()

    dense_embeddings = get_clip_embeddings_dense(image)
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
