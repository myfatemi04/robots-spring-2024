from typing import List, Tuple

import PIL.Image
import torch
from transformers import SamModel, SamProcessor

# to create a better mask, we'll use the bounding boxes as prompts for SAM, which can try to exclude some of the background
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model: SamModel = SamModel.from_pretrained("facebook/sam-vit-base").to(device) # type: ignore
sam_processor: SamProcessor = SamProcessor.from_pretrained("facebook/sam-vit-base") # type: ignore

def boxes_to_masks(image: PIL.Image.Image, input_boxes: List[Tuple[int, int, int, int]]):
    with torch.no_grad():
        inputs = sam_processor(images=[image], input_boxes=[[list(box) for box in input_boxes]], return_tensors="pt").to(device)
        outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks( # type: ignore
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),      # type: ignore
            inputs["reshaped_input_sizes"].cpu() # type: ignore
        )[0] # 0 to remove batch dimension
        return [mask[0].detach().cpu().numpy().astype(float) for mask in masks]

def points_to_masks(image: PIL.Image.Image, points: List[Tuple[int, int]]):
    masks = []
    with torch.no_grad():
        i = 0
        while i < len(points):
            inputs = sam_processor(images=[image], input_points=[[[list(pt)] for pt in points[i:i + 1]]], return_tensors="pt").to(device)
            outputs = sam_model(**inputs)
            masks.extend([m[0].detach().cpu().numpy().astype(bool) for m in sam_processor.image_processor.post_process_masks( # type: ignore
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),      # type: ignore
                inputs["reshaped_input_sizes"].cpu() # type: ignore
            )[0]]) # 0 to remove image batch dimension
            i += 1
    return masks
