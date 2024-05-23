import io
from typing import List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import PIL.Image as Image

from .detect_objects import Detection


def draw_set_of_marks(image: Image.Image, predictions: List[Detection], custom_labels=None):
    fig = plt.figure(figsize=(8, 6), dpi=128)
    ax = fig.add_subplot(111)
    
    ax.imshow(image)

    object_id_counter = 1
    for prediction in predictions:
        if type(prediction) in [tuple, list]:
            x1, y1, x2, y2 = prediction
        elif isinstance(prediction, Detection):
            x1, y1, x2, y2 = prediction.box
        else:
            raise TypeError("Invalid type passed in for prediction list")
        
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

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=128, bbox_inches='tight')
    buf.seek(0)
    plt.clf()

    return Image.open(buf)