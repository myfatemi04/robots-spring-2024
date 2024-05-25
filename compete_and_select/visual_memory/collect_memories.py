import json
import os

import numpy as np
import PIL.Image
import PIL.ImageFilter
import torch
from matplotlib import pyplot as plt

from ..object_detection import detect_objects_few_shot as D
from ..object_detection.detect_objects_few_shot import (ImageObservation,
                                                        boxes_to_masks,
                                                        select_bounding_box)


# Store memories to disk.
def collect_memories_for_image(img):
    label_more = True
    groups = []
    while label_more:
        description = input("Description:")
        bboxes = [select_bounding_box(img)]
        while 'y' == input('More boxes? (y/n)'):
            bboxes.append(select_bounding_box(img))

        groups.append({'description': description, 'bboxes': bboxes})

        label_more = 'y' == input('More facts? (y/n)')

    # Store these to disk.
    counter = 0
    while os.path.exists(f"memories/{counter}"):
        counter += 1
    os.makedirs(f"memories/{counter}")
    img.save(f"memories/{counter}/img.png")
    with open(f"memories/{counter}/groups.json", "w") as f:
        json.dump(groups, f)

def collect_memories_live():
    from compete_and_select.perception.rgbd import RGBD

    rgbd = RGBD(num_cameras=1)
    try:
        (rgbs, pcds) = rgbd.capture()
        imgs = [PIL.Image.fromarray(rgb) for rgb in rgbs]
        collect_memories_for_image(imgs[0])
        
    except Exception as e:
        print("Error: ", e)
    finally:
        rgbd.close()

def fix_size(img, groups):
    # resize image to max height of 448
    if img.height > 448:
        scale = 448 / img.height
        img = img.resize((int(img.width * 448 / img.height), 448))
        groups = [
            {"description": group['description'], 'bboxes': [[a * scale for a in bbox] for bbox in group['bboxes']]}
            for group in groups
        ]
    # Resize image to be a multiple of 14
    w = img.width - (img.width % 14)
    h = img.height - (img.height % 14)
    img = img.crop((0, 0, w, h))
    return img, groups

def load_memories(folder):
    with open(f"{folder}/groups.json", "r") as f:
        groups = json.load(f)
    img = PIL.Image.open(f"{folder}/img.png")
    img, groups = fix_size(img, groups)
    img_blurred = img.filter(PIL.ImageFilter.GaussianBlur(5))

    img_O = ImageObservation(img)
    img_blurred_O = ImageObservation(img_blurred)

    from compete_and_select.perception.rgbd import RGBD
    rgbd = RGBD(num_cameras=1)
    (rgbs, pcds) = rgbd.capture()
    rgbd.close()
    new_img = PIL.Image.fromarray(rgbs[0])
    new_img, _= fix_size(new_img, [])
    new_img_O = ImageObservation(new_img)

    groups = [
        {'description': 'candy', 'bboxes': [groups[i]['bboxes'][0] for i in range(len(groups))]}
    ]

    for group in groups:
        masks = boxes_to_masks(img, group['bboxes'])
        combined_mask = np.sum(masks, axis=0) > 0
        # os.system("nvidia-smi")

        plt.imshow(img)
        plt.imshow(combined_mask, alpha=combined_mask.astype(float))
        plt.show()
        # os.system("nvidia-smi")

        svc = D.compile_memories([img_O, img_blurred_O], [combined_mask, combined_mask])

        # os.system("nvidia-smi")
        
        # now, we have a new image (for example)
        highlight = D.highlight(new_img_O, svc)
        highlight = PIL.Image.fromarray((highlight * 255).astype(np.uint8))
        highlight = highlight.resize((new_img.width, new_img.height))
        highlight = np.array(highlight) / 255

        # draw
        plt.imshow(new_img)
        plt.imshow(highlight, alpha=highlight)
        plt.show()
        
        torch.cuda.empty_cache()

        kept_masks = D.create_masks_from_highlight(highlight, new_img_O.image)
        
        # draw
        plt.imshow(new_img)
        plt.imshow(highlight, alpha=highlight)
        for mask in kept_masks:
            plt.imshow(mask, alpha=mask.astype(float))
        plt.show()

# collect_memories_for_image(PIL.Image.open("sample_images/IMG_8618.jpeg"))
load_memories("memories/0")

