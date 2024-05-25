from matplotlib import pyplot as plt
import numpy as np
from transformers import pipeline

from .. import RGBD

generator = pipeline("mask-generation", model="facebook/sam-vit-base", device=0)

rgbd = RGBD.autoload('front_only_color')
imgs, pcds = rgbd.capture(return_pil=True)

masks = generator(imgs[0], points_per_batch=64, points_per_crop=64)['masks']

plt.title("Image with Masks")
plt.imshow(imgs[0])

for mask in masks:
    color = np.random.rand(4)
    color[3] = 0.5
    mask_image = mask.reshape(mask.shape[0], mask.shape[1], 1).astype(float)
    mask_image = mask_image.repeat(4, axis=2)
    mask_image[mask] = color
    mask_image[~mask][:, 3] = 0
    plt.imshow(mask_image)

plt.show()
