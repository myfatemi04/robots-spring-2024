import os
import time
from matplotlib import pyplot as plt

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

import sys
sys.path.append("../../Cutie")

# Note that you may need to install hydra-core to do this.

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

sys.path.pop()

from rgbd import RGBD
from select_bounding_box import select_bounding_box
from sam import boxes_to_masks
import PIL.Image

# @torch.cuda.amp.autocast()
@torch.inference_mode()
def main():
    # obtain the Cutie model with default parameters -- skipping hydra configuration
    cutie = get_default_model()
    # Typically, use one InferenceCore per video
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    rgbd = RGBD(num_cameras=1)
    rgbs, pcds = rgbd.capture()

    # Create a segmentation mask.
    image = PIL.Image.fromarray(rgbs[0])
    bbox = select_bounding_box(image)
    mask = torch.from_numpy(boxes_to_masks(image, [bbox])[0]).cuda()

    image_pil = image
    image = to_tensor(image).cuda().float()
    # If an object is not in the `objects` parameter, it gets propagated using memory.
    processor.step(image, mask, objects=[1])

    plt.title("Initial Mask")
    plt.imshow(image_pil)
    plt.imshow(mask.cpu().numpy(), alpha=mask.cpu().numpy())
    plt.show()

    mask = None
    
    ti = 0
    start_time = time.time()
    while time.time() < start_time + 30:
        rgbs, pcds = rgbd.capture()
        # load the image as RGB; normalization is done within the model
        image = PIL.Image.fromarray(rgbs[0])
        image_pil = image
        image = to_tensor(image).cuda().float()

        # This is how we would delete an object.
        # processor.delete_objects([1])

        # we pass the mask in if it exists
        if False:
            # if mask is passed in, it is memorized
            # if not all objects are specified, we propagate the unspecified objects using memory
            objects = torch.unique(mask)
            objects = objects[objects != 0].tolist()
            output_prob = processor.step(image, mask, objects=objects)
        else:
            # otherwise, we propagate the mask from memory
            output_prob = processor.step(image)

        # convert output probabilities to an object mask
        mask = processor.output_prob_to_mask(output_prob).cpu().numpy()

        # Yes, we could just cast to a float directly, but want to be explicit
        # about what we are doing here.
        mask = (mask == 1).astype(float)

        plt.title("Tracked Mask")
        plt.imshow(image_pil)
        plt.imshow(mask, alpha=mask)
        plt.pause(0.05)

        # Note that we can also create images using Image.fromarray(mask)
        # and mask_img.putpalette(palette).

main()