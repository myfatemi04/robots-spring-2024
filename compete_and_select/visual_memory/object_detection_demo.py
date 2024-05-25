import numpy as np
import PIL.Image
import PIL.ImageFilter
import torch
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from sklearn.svm import SVC

from ..clip_feature_extraction import (get_full_scale_clip_embedding_tiles,
                                       get_full_scale_clip_embeddings)
from ..detect_objects import detect
from ..perception.rgbd import RGBD
from ..sam import boxes_to_masks
from ..select_bounding_box import select_bounding_box


def show_detections_live():
    rgbd = RGBD(num_cameras=1)

    prompt = input("What should I look for: ")

    try:
        while True:
            (rgbs, pcds) = rgbd.capture()
            img = PIL.Image.fromarray(rgbs[0])
            detections = detect(img, prompt)

            plt.clf()
            plt.axis('off')
            plt.imshow(img)

            ax = plt.gca()

            for detection in detections:
                rect = mpatches.Rectangle(
                    (detection.box[0], detection.box[1]),
                    detection.box[2] - detection.box[0],
                    detection.box[3] - detection.box[1],
                    fill=False,
                    edgecolor='red',
                    linewidth=2,
                )
                # ax.add_patch(rect)
                ax.text(detection.box[0], detection.box[1], f"{detection.score:.2f}", color='red')

            plt.pause(0.05)

    except Exception as e:
        print("Error: ", e)

    finally:
        rgbd.close()

def choice(array, count):
    return array[torch.randperm(len(array))[:count]]

class Memory:
    def __init__(self):
        self.images = []
        self.masks = []
        
        self._images = []
        self._masks = []
        self._embeds = []

        self._positive_examples = []
        self._negative_examples = []
        
    def add(self, img, mask):
        self.images.append(img)
        self.masks.append(mask)

        # Generate augmented images.
        blur = PIL.ImageFilter.GaussianBlur(5)

        self._images.append(img)
        self._images.append(img.filter(blur))
        self._masks.append(mask)
        self._masks.append(mask)

        img_array = np.array(img)
        blank_image = np.ones_like(img_array) * 255
        blank_image[mask] = img_array[mask]
        self._images.append(PIL.Image.fromarray(blank_image))
        self._masks.append(mask)

        # Generate positive and negative examples.
        for image in self._images:
            clip_field = get_full_scale_clip_embeddings(image)['rescaled_cropped']
            self._embeds.append(clip_field)

        for embed, mask in zip(self._embeds, self._masks):
            self._positive_examples.append(choice(embed[mask], 10))
            self._negative_examples.append(choice(embed[~mask], 10))

    def generate_svm(self):
        pos = np.concatenate(self._positive_examples)
        neg = np.concatenate(self._negative_examples)
        examples = np.concatenate([pos, neg], axis=0)
        labels = np.array([1] * len(pos) + [0] * len(neg))

        # Train an SVM.
        svm = SVC(kernel='linear', probability=True).fit(examples, labels)

        return svm
    
    def infer(self, features):
        return self.generate_svm() \
            .predict_proba(features.reshape(-1, 768))[:, 1] \
            .reshape(features.shape[:2])

def collect_samples():
    rgbd = RGBD(num_cameras=1)

    rgbs, pcds = rgbd.capture()
    img = PIL.Image.fromarray(rgbs[0])

    # User selects an object.
    bbox = select_bounding_box(img)
    mask = boxes_to_masks(img, [bbox])[0]

    # Visualize the mask.
    plt.clf()
    plt.axis('off')
    plt.title("Mask")
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5 * mask.astype(float))
    plt.show()
    
    mem = Memory()
    mem.add(img, mask)

    # Apply again to the original image.
    clip_discrete_field = get_full_scale_clip_embedding_tiles(img)

    probs = mem.infer(clip_discrete_field)
    probs = np.array(PIL.Image.fromarray(probs).resize(img.size, resample=PIL.Image.NEAREST))

    # Take anything positively-classified, that is not in the mask, and remake the dataset.
    false_positives = (probs > 0.5) & (~mask)
    false_positive_embeds = choice(mem._embeds[0][false_positives], 10)

    # visualize.
    plt.clf()
    plt.axis('off')
    plt.title("False positives")
    plt.imshow(img)
    plt.imshow(false_positives, alpha=false_positives.astype(float))
    plt.show()

    pos = np.concatenate(mem._positive_examples)
    neg = np.concatenate(mem._negative_examples)
    print(pos.shape, neg.shape, false_positive_embeds.shape)
    examples = np.concatenate([
        pos,
        neg,
        false_positive_embeds
    ], axis=0)
    labels = np.array([1] * len(pos) + [0] * (len(neg) + len(false_positive_embeds)))

    # Train an SVM.
    svm = SVC(kernel='linear', probability=True).fit(examples, labels)

    try:
        # Evaluate the SVM on new examples.
        while True:
            rgbs, pcds = rgbd.capture()
            img = PIL.Image.fromarray(rgbs[0])

            clip_discrete_field = get_full_scale_clip_embedding_tiles(img)

            probs = svm \
                .predict_proba(clip_discrete_field.reshape(-1, 768))[:, 1] \
                .reshape(clip_discrete_field.shape[:2])
            
            probs = np.array(PIL.Image.fromarray(probs).resize(img.size, resample=PIL.Image.NEAREST))

            plt.clf()
            plt.axis('off')
            plt.imshow(img)
            plt.imshow(probs, alpha=probs, cmap='coolwarm', vmin=0, vmax=1)
            
            plt.pause(0.05)

    except Exception as e:
        print("Error: ", e)

    finally:
        rgbd.close()

collect_samples()
