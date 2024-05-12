import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.ops as ops
from object_detection_utils import add_object_clip_embeddings
from transformers import Owlv2ForObjectDetection, Owlv2Processor

model_name = 'google/owlv2-large-patch14-ensemble'
# Faster
# model_name = "google/owlv2-base-patch16-ensemble"
processor = Owlv2Processor.from_pretrained(model_name)
model = Owlv2ForObjectDetection.from_pretrained(model_name).to("cuda")

@dataclass
class Detection:
    box: Tuple[float, float, float, float]
    score: float
    label: str
    embedding: Optional[np.ndarray]

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

def _fix_detection_heights(predictions, image_width, image_height):
    predictions_2 = []
    height_scale = image_width / image_height
    for prediction in predictions:
        box = prediction['box']
        # Needed for OWL-ViT for some reason.
        rescaled_prediction = {
            'box': {
                'xmin': box['xmin'],
                'ymin': box['ymin'] * height_scale,
                'xmax': box['xmax'],
                'ymax': box['ymax'] * height_scale,
            },
            'score': prediction['score'],
            'label': prediction['label'],
        }
        predictions_2.append(rescaled_prediction)
    return predictions_2

def detect(image, label):
    start = time.time()

    with torch.no_grad():
        inputs = processor(text=[label.lower()], images=image, return_tensors="pt").to("cuda")
        outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    # Corresponds to texts[0], which is just label
    results = results[0]
    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
    predictions = [{"box": {
        'xmin': box[0].item(),
        'ymin': box[1].item(),
        'xmax': box[2].item(),
        'ymax': box[3].item(),
    }, "score": score, "label": label} for (box, score, label) in zip(boxes, scores, labels)]

    end = time.time()

    print(f"Detection duration: {end - start:.2f}")
    
    predictions = _fix_detection_heights(predictions, image.width, image.height)

    boxes_xy = []
    for prediction in predictions:
        box = prediction['box']
        boxes_xy.append([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
    boxes_xy = torch.tensor(boxes_xy).float().reshape(-1, 4) # in case no boxes are found, give it shape [0, 4]
    box_sizes = (boxes_xy[:, 3] - boxes_xy[:, 1]) * (boxes_xy[:, 2] - boxes_xy[:, 0]) / (image.height * image.width)

    scores = torch.tensor([prediction['score'] for prediction in predictions])

    keep = ops.nms(boxes_xy, scores, iou_threshold=0.2)
    keep = {int(i) for i in keep}
    # do not accept boxes that fill a whole quadrant of space
    keep = {i for i in keep if box_sizes[i] < 0.25}

    detections = add_object_clip_embeddings(image, [
        prediction for (i, prediction) in enumerate(predictions) if i in keep
    ])

    return [
        Detection(
            box=(detection['box']['xmin'], detection['box']['ymin'], detection['box']['xmax'], detection['box']['ymax']),
            score=detection['score'],
            label=detection['label'],
            embedding=detection['emb'],
        )
        for detection in detections
    ]

def detect_v0(image, label):
    start = time.time()
    with torch.no_grad():
        inputs = processor(text=[label.lower()], images=image, return_tensors="pt").to("cuda")
        outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    # Corresponds to texts[0], which is just label
    results = results[0]
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
    predictions = [{"box": {
        'xmin': box[0].item(),
        'ymin': box[1].item(),
        'xmax': box[2].item(),
        'ymax': box[3].item(),
    }, "score": score, "label": label} for (box, score, label) in zip(boxes, scores, labels)]

    # predictions = detector(
    #     image,
    #     candidate_labels=[label],
    # )
    end = time.time()

    print(f"Detection duration: {end - start:.2f}")
    
    predictions_2 = []
    height_scale = image.width / image.height
    for prediction in predictions:
        box = prediction['box']
        # Needed for OWL-ViT for some reason.
        rescaled_prediction = {
            'box': {
                'xmin': box['xmin'],
                'ymin': box['ymin'] * height_scale,
                'xmax': box['xmax'],
                'ymax': box['ymax'] * height_scale,
            },
            'score': prediction['score'],
            'label': prediction['label'],
        }
        predictions_2.append(rescaled_prediction)
    predictions = predictions_2

    boxes_xy = []
    for prediction in predictions:
        box = prediction['box']
        boxes_xy.append([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
    boxes_xy = torch.tensor(boxes_xy).float().reshape(-1, 4) # in case no boxes are found, give it shape [0, 4]
    box_sizes = (boxes_xy[:, 3] - boxes_xy[:, 1]) * (boxes_xy[:, 2] - boxes_xy[:, 0]) / (image.height * image.width)

    scores = torch.tensor([prediction['score'] for prediction in predictions])

    keep = ops.nms(boxes_xy, scores, iou_threshold=0.2)
    keep = {int(i) for i in keep}
    # do not accept boxes that fill a whole quadrant of space
    keep = {i for i in keep if box_sizes[i] < 0.25}

    return add_object_clip_embeddings(image, [
        prediction
        for (i, prediction) in enumerate(predictions)
        if i in keep
    ])
