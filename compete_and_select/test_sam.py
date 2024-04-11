import transformers

from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import torch

model = SamModel.from_pretrained("facebook/sam-vit-base").cuda()
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]] # 2D localization of a window

with torch.no_grad():
    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores

import matplotlib.pyplot as plt

plt.imshow(raw_image)
for i in range(len(masks[0][0])):
    plt.imshow(masks[0][0][i], alpha=masks[0][0][i] * scores[0, 0, i].cpu())

plt.axis('off')
plt.title("Detections")
plt.show()
