import sys
import time

import detect_objects_few_shot as D
import PIL.Image
import PIL.ImageFilter
import torch
from matplotlib import pyplot as plt
from rgbd import RGBD
from sam import boxes_to_masks
from select_bounding_box import select_bounding_box
from torchvision.transforms.functional import to_tensor

### Import Cutie ###
sys.path.append("../../Cutie")
# Note that you may need to install hydra-core to do this.
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

sys.path.pop()

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

    # Resize image to max height of 448
    width = image.width
    height = image.height
    new_height = 448
    new_width = int(width * new_height / height)
    new_width -= (new_width % 14)
    image = image.resize((new_width, new_height))

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

    images = []
    masks = []
    
    ti = 0
    start_time = time.time()
    while ti < 10:
        print(ti)
        rgbs, pcds = rgbd.capture()
        # load the image as RGB; normalization is done within the model
        image_pil = PIL.Image.fromarray(rgbs[0])
        image_pil = image_pil.resize((new_width, new_height))
        image = to_tensor(image_pil).cuda().float()

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

        images.append(image_pil)
        masks.append(mask)

        plt.title("Tracked Mask")
        plt.imshow(image_pil)
        plt.imshow(mask, alpha=mask)
        plt.pause(0.05)

        # Note that we can also create images using Image.fromarray(mask)
        # and mask_img.putpalette(palette).
        ti += 1

    # Create a set of positive and negative examples.
    images = [D.ImageObservation(img) for img in images]
    blur = PIL.ImageFilter.GaussianBlur(2)
    for i in range(len(images)):
        images.append(D.ImageObservation(images[i].image.filter(blur)))
        masks.append(masks[i])
    svc = D.compile_memories(images, masks)

    # Now, reopen the camera, and highlight where the object is.
    start_time = time.time()
    while time.time() - start_time <= 30:
        rgbs, pcds = rgbd.capture()
        image_pil = PIL.Image.fromarray(rgbs[0])
        image_pil = image_pil.resize((new_width, new_height))
        image_obs = D.ImageObservation(image_pil)
        
        hl = D.highlight(image_obs, svc)
        D.visualize_highlight(image_obs, hl, plt_pause=0.05)

main()