"""
Here, we label photos with bounding boxes.
"""

import json
import os
import pickle

from matplotlib import pyplot as plt
from PIL import Image

from .class_labels import class_labels
from ..sam import boxes_to_masks
from ..select_bounding_box import select_bounding_box


def get_all_classes(filenames):
    classes = set()
    for filename in filenames:
        basename = os.path.splitext(filename)[0]
        if 'no_' in basename:
            parts = basename.split('_')[2:]  # Get all parts after "no_"
            class_name = '_'.join(parts)
            classes.add(class_name)
    return list(classes)

def extract_classes(filename, all_classes):
    basename = os.path.splitext(filename)[0]
    if 'no_' in basename:
        parts = basename.split('_')[2:]  # Get all parts after "no_"
        excluded_class = '_'.join(parts)
        classes = [cls for cls in all_classes if cls != excluded_class]
    else:
        classes = all_classes
    return sorted(classes)

def load_images_and_classes(prefix_dir):
    filenames = sorted(os.listdir(prefix_dir))
    # filter out a labels.json if it's in there, or potentially a masks.pkl.
    filenames = [filename for filename in filenames if '.json' not in filename and '.pkl' not in filename]
    all_classes = sorted(get_all_classes(filenames))
    image_class_tuples = []
    for filename in filenames:
        # Load the image
        image = Image.open(os.path.join(prefix_dir, filename))
        # Extract the classes
        classes = extract_classes(filename, all_classes)
        # Append the tuple to the list
        image_class_tuples.append((filename, image, classes))
    return image_class_tuples

def label_bounding_boxes(prefix_dir):
    # Execute the function with the list of filenames
    image_class_tuples = load_images_and_classes(prefix_dir)

    # Save the list of expected labels per image
    expected = {
        filename: classes
        for filename, _, classes in image_class_tuples
    }
    with open(os.path.join(prefix_dir, 'expected.json'), 'w') as f:
        json.dump(expected, f)

    all_labels = {filename: None for filename, _, _ in image_class_tuples}

    # Print the results for verification
    for filename, image, classes in image_class_tuples:
        print(f"Image: {filename}, Classes: {classes}")

        labels = []

        for cls in classes:
            bbox = select_bounding_box(image, "Select the bounding box for the class: " + cls)
            labels.append((cls, bbox))
            print("Labeled class", cls, "with bounding box:", bbox)

        all_labels[filename] = labels # type: ignore

    with open(os.path.join(prefix_dir, 'labels.json'), 'w') as f:
        json.dump(all_labels, f)

def generate_masks_from_bounding_boxes(prefix_dir, labels):
    mask_results = {}

    for filename, labels in labels.items():
        print("Showing masks for filename", filename)
        image = Image.open(os.path.join(prefix_dir, filename))

        labels = sorted(labels, key=lambda x: x[0])

        # should be a Boolean ndarray
        masks = boxes_to_masks(image, [bbox for _, bbox in labels])
        for (cls, bbox), mask in zip(labels, masks):
            plt.title("Generated mask: " + cls)
            plt.imshow(image)
            plt.imshow(mask.astype(float), alpha=mask.astype(float))
            plt.show()

        all_ok = 'y' == input('All OK? (y/n) ')
        if not all_ok:
            print("OK mask files:")
            print(list(mask_results.keys()))
            print("Current filename:", filename)
            print("... Exiting so the data can be corrected ...")
            break
        
        mask_results[filename] = masks
        
    with open(os.path.join(prefix_dir, 'masks.pkl'), 'wb') as f:
        pickle.dump(mask_results, f)

if __name__ == "__main__":
    # 1. Generate bounding boxes
    # 2. Generate masks

    """
    Groups completed:
     - Condiments (coarse)
     - Cups (coarse)
     - Spoons (coarse)

    Groups to complete:
     - Cups (direct)
     - Spoons (direct)
     - Condiments (direct)
     - Canisters (direct)
     - Canisters (coarse)
     - Writing utensils (direct)
     - Writing utensils (coarse)
    """

    group = 'writing_utensils'

    prefix_dir = './photos/solidbg/' + group
    slug_labels = class_labels[group]['slug_labels']
    natural_language_labels = class_labels[group]['natural_language_labels']

    label_bounding_boxes(prefix_dir)

    json_path = os.path.join(prefix_dir, 'labels.json')
    with open(json_path, 'r') as f:
        labels = json.load(f)

    print("All labels:")
    print([cls for (cls, bbox) in labels['000_all.png']])

    generate_masks_from_bounding_boxes(prefix_dir, labels)

    # print(
    #     select_bounding_box(
    #         Image.open('./photos/solidbg/cups/000_all.png'),
    #         "Select coffee cup bruh"
    #     )
    # )
