from matplotlib import pyplot as plt
import numpy as np
from ..perception.rgbd import RGBD
from ..clip_feature_extraction import get_full_scale_clip_embedding_tiles, get_text_embeds
import PIL.Image

rgbd = RGBD.autoload('front_only_color')

# We normalize embeddings, as is done in the original CLIP paper.
# This essentially uses cosine similarity, as the dot product between
# two vectors is the cosine of the angle between them times the product
# of their magnitudes.
# text_embed = get_text_embeds('a photo of a cup')[0]
text_embed = get_text_embeds('a photo of candy')[0]
text_embed /= np.linalg.norm(text_embed)

try:
    while True:
        imgs, pcds = rgbd.capture(return_pil=True)
        image = imgs[0]
        # image = image.resize((224, int(224 * (image.height / image.width))), resample=PIL.Image.BILINEAR)

        tiles = get_full_scale_clip_embedding_tiles(image)
        tiles /= np.linalg.norm(tiles, axis=-1, keepdims=True)
        scores = tiles @ text_embed
        scores -= np.min(scores)
        scores /= np.max(scores)
        scores_image = PIL.Image.fromarray(scores).resize((image.width, image.height), resample=PIL.Image.NEAREST)
        scores_image = np.array(scores_image)
        
        plt.clf()
        plt.title("CLIP Match Score")
        plt.axis('off')
        plt.imshow(image)
        plt.imshow(scores_image, alpha=scores_image, vmin=0, vmax=1, cmap='coolwarm')
        plt.pause(0.05)

finally:
    rgbd.close()
