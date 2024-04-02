import io
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torchvision.ops as ops
from PIL import Image

# Transformers 4.33 doesn't have OwlViT as a pipeline, but we can use it manually.
from transformers import pipeline

checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device="cuda")
# from transformers import OwlViTProcessor, OwlViTForObjectDetection

# processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
# model = OwlViTForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to("cuda")

def draw_set_of_marks(image, predictions, custom_labels=None, live=False):
    plt.clf()

    # plt.title("Results from OWL-ViT")
    plt.imshow(image)

    object_id_counter = 1
    for prediction in predictions:
        box = prediction["box"]
        # label = prediction["label"]
        # score = prediction["score"]
        
        x1 = box['xmin']
        x2 = box['xmax']
        y1 = box['ymin']
        y2 = box['ymax']
        
        plt.gca().add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor='none', edgecolor='r', linewidth=2))
        
        text_x = x1
        text_y = max(y1 - 20, 10)
        
        if (x1 / image.width) > 0.9:
            text_x = x2
            horizontalalignment = 'right'
        else:
            horizontalalignment = 'left'
            
        plt.text(
            text_x,
            text_y,
            f"#{object_id_counter}" if custom_labels is None else custom_labels[object_id_counter - 1],
            c='white',
            backgroundcolor=(0.1, 0.1, 0.1, 0.5),
            horizontalalignment=horizontalalignment
        )
        plt.axis('off')
        
        object_id_counter += 1

    plt.tight_layout()

    if not live:
        # Save to PIL image.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        plt.clf()

        return Image.open(buf)
    else:
        # plt.show()
        # let the caller do show()
        pass

def detect(image, label):
    start = time.time()
    # inputs = processor(text=[label], images=image, return_tensors="pt").to("cuda")
    # outputs = model(**inputs)

    # # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    # target_sizes = torch.Tensor([image.size[::-1]])
    # # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    # results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    # # Corresponds to texts[0], which is just label
    # results = results[0]
    # i = 0  # Retrieve predictions for the first image for the corresponding text queries
    # boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
    # predictions = [{"box": box, "score": score, "label": label} for (box, score, label) in zip(boxes, scores, labels)]

    predictions = detector(
        image,
        candidate_labels=[label],
    )
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

    return [prediction for (i, prediction) in enumerate(predictions) if i in keep]
