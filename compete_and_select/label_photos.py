"""
Here, we label photos with bounding boxes.
"""

import os
import json

from PIL import Image

from .select_bounding_box import select_bounding_box


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
    return classes

def load_images_and_classes(prefix_dir):
    filenames = sorted(os.listdir(prefix_dir))
    # filter out a labels.json if it's in there, or potentially a masks.pkl.
    filenames = [filename for filename in filenames if '.json' not in filename and '.pkl' not in filename]
    all_classes = get_all_classes(filenames)
    image_class_tuples = []
    for filename in filenames:
        # Load the image
        image = Image.open(os.path.join(prefix_dir, filename))
        # Extract the classes
        classes = extract_classes(filename, all_classes)
        # Append the tuple to the list
        image_class_tuples.append((filename, image, classes))
    return image_class_tuples

def main():
    # Assuming images are in the current directory
    prefix_dir = './photos/solidbg/cups'

    # Execute the function with the list of filenames
    image_class_tuples = load_images_and_classes(prefix_dir)

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

if __name__ == "__main__":
    main()
