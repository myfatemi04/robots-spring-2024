from PIL import Image
import requests
from transformers import AutoProcessor, CLIPSegForImageSegmentation
import torchvision.transforms.functional as TF
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device) # type: ignore

def detect_objects_clipseg(image, object, resize_output=True):
    inputs = clipseg_processor(text=[object], images=[image], padding=True, return_tensors="pt").to(device)

    outputs = clipseg_model(**inputs)

    # Resize.
    width = image.width
    height = image.height
    
    if resize_output:
        logits = TF.resize(outputs.logits.unsqueeze(0), [height, width]).squeeze(0)
        
    logits = TF.resize(outputs.logits.unsqueeze(0), [height, width]).squeeze(0)
    logits = logits.detach().cpu().numpy()
    
    return logits
