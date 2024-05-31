import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
vlm = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    revision=revision,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)