import numpy as np
import PIL.Image
from matplotlib import cm, patches
from matplotlib import pyplot as plt
import torch
from transformers import pipeline

from .. import RGBD
from ..clip_feature_extraction import (get_full_scale_clip_embedding_tiles,
                                       get_text_embeds)
from .describe_objects import describe_objects

from torchvision.ops import masks_to_boxes

generator = pipeline("mask-generation", model="facebook/sam-vit-base", device=0)

rgbd = RGBD.autoload('front_only_color')
imgs, pcds = rgbd.capture(return_pil=True)
image = imgs[0]

# See `clip_segmentation_demo.py` for why we normalize these.
text_embed = get_text_embeds('a photo of candy')[0]
text_embed /= np.linalg.norm(text_embed)

masks = generator(image, points_per_batch=64, points_per_crop=32)['masks']
masks = torch.tensor(np.stack(masks, axis=0))

# Get bounding boxes for the masks.
bounding_boxes = masks_to_boxes(masks)

print(len(bounding_boxes), "objects detected.")

plt.title("Image with Masks")
plt.imshow(image)

tiles = get_full_scale_clip_embedding_tiles(image)
tiles /= np.linalg.norm(tiles, axis=-1, keepdims=True)
scores = tiles @ text_embed

# Optional: Min-Max Scaling
scores -= scores.min()
scores /= scores.max()

scores_image = PIL.Image.fromarray(scores).resize((image.width, image.height), resample=PIL.Image.BILINEAR)
scores_image = np.array(scores_image)

for (bbox, mask) in zip(bounding_boxes, masks):
    score = scores_image[mask].mean()
    color = np.array([*cm.coolwarm(score)[:3], 0.9])

    # Use a very moderate threshold
    if score > 0.5:
        color[0] = 1.0
    else:
        # color[2] = 1.0
        color[3] = 0.0

    mask_image = mask.reshape(mask.shape[0], mask.shape[1], 1).numpy().astype(float)
    mask_image = mask_image.repeat(4, axis=2) * color
    plt.imshow(mask_image)

    x1, y1, x2, y2 = bbox

    plt.gca().add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none'))

plt.show()
