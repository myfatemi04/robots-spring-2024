from prismatic import load
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
# model_id = "phi-2+3b"
model_id = "train-2-epochs+7b"
vlm = load(model_id)
vlm.to(device, dtype=torch.bfloat16)
