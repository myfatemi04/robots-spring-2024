import io
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torchvision.ops as ops
from PIL import Image

from transformers import Owlv2Processor, Owlv2ForObjectDetection, CLIPVisionModelWithProjection, CLIPProcessor

model_name = 'google/owlv2-large-patch14-ensemble'
# Faster
# model_name = "google/owlv2-base-patch16-ensemble"
processor = Owlv2Processor.from_pretrained(model_name)
model = Owlv2ForObjectDetection.from_pretrained(model_name).to("cuda")

clip_vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# create clip embeddings for objects
def add_object_clip_embeddings(im1, dets1):
    with torch.no_grad():
        for det in dets1:
            box = det['box']
            width = box['xmax'] - box['xmin']
            height = box['ymax'] - box['ymin']
            center_x = (box['xmax'] + box['xmin']) / 2
            center_y = (box['ymax'] + box['ymin']) / 2
            # ensure square image to prevent warping
            size = max(224, width, height)
            object_img = im1.crop((center_x-size/2, center_y-size/2, center_x+size/2, center_y+size/2))
            object_emb_output = clip_vision_model(**clip_processor(images=[object_img], return_tensors='pt').to('cuda'))
            object_emb = object_emb_output.image_embeds[0]

            det['emb'] = object_emb
            # det['desc'] = image_to_text(object_img)
        
    return dets1

def draw_set_of_marks(image, predictions, custom_labels=None, live=False):
    fig = plt.figure(figsize=(8, 6), dpi=128)
    ax = fig.add_subplot(111)
    
    ax.imshow(image)

    object_id_counter = 1
    for prediction in predictions:
        box = prediction["box"]
        # label = prediction["label"]
        # score = prediction["score"]
        
        x1 = box['xmin']
        x2 = box['xmax']
        y1 = box['ymin']
        y2 = box['ymax']
        
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor='none',
            edgecolor='r',
            linewidth=2
        ))
        
        text_x = x1
        text_y = max(y1 - 15, 10)
        
        if (x1 / image.width) > 0.9:
            text_x = x2
            horizontalalignment = 'right'
        else:
            horizontalalignment = 'left'
            
        ax.text(
            text_x,
            text_y,
            str(object_id_counter) if custom_labels is None else custom_labels[object_id_counter - 1],
            c='white',
            backgroundcolor=(0, 0, 0, 1.0),
            horizontalalignment=horizontalalignment,
            size=10,
        )
        
        object_id_counter += 1

    ax.axis('off')
    fig.tight_layout()

    if not live:
        # Save to PIL image.
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=128, bbox_inches='tight')
        buf.seek(0)
        plt.clf()

        return Image.open(buf)
    # let the caller do .show()

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
