import os
from matplotlib import pyplot as plt
from detect_objects_few_shot import select_bounding_box, ImageObservation, boxes_to_masks
import PIL.Image
import json

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
    from rgbd import RGBD

    rgbd = RGBD(num_cameras=1)
    try:
        (rgbs, pcds) = rgbd.capture()
        imgs = [PIL.Image.fromarray(rgb) for rgb in rgbs]
        collect_memories_for_image(imgs[0])
        
    except Exception as e:
        print("Error: ", e)
    finally:
        rgbd.close()

def load_memories(folder):
    with open(f"{folder}/groups.json", "r") as f:
        groups = json.load(f)
    img = PIL.Image.open(f"{folder}/img.png")
    img_blurred = img.filter(PIL.ImageFilter.GaussianBlur(5))
    masks = boxes_to_masks(img, [group['bboxes'] for group in groups])

    ImageObservation(img)
    ImageObservation(img_blurred)

# collect_memories_for_image(PIL.Image.open("sample_images/IMG_8618.jpeg"))

