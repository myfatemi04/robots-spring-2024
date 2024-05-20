import torch
from transformers import SamModel, SamProcessor

# to create a better mask, we'll use the bounding boxes as prompts for SAM, which can try to exclude some of the background
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model: SamModel = SamModel.from_pretrained("facebook/sam-vit-base").to(device) # type: ignore
sam_processor: SamProcessor = SamProcessor.from_pretrained("facebook/sam-vit-base") # type: ignore
