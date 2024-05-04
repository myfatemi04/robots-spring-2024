import io

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

clip_vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# create clip embeddings for objects
def add_object_clip_embeddings(image, detections):
    with torch.no_grad():
        for detection in detections:
            box = detection['box']
            width = box['xmax'] - box['xmin']
            height = box['ymax'] - box['ymin']
            center_x = (box['xmax'] + box['xmin']) / 2
            center_y = (box['ymax'] + box['ymin']) / 2
            # ensure square image to prevent warping
            size = max(224, width, height)
            object_img = image.crop((center_x-size/2, center_y-size/2, center_x+size/2, center_y+size/2))
            object_emb_output = clip_vision_model(**clip_processor(images=[object_img], return_tensors='pt').to('cuda'))
            object_emb = object_emb_output.image_embeds[0]

            detection['emb'] = object_emb
        
    return detections

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