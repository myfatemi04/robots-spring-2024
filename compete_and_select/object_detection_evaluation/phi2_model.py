from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

### Final Verdict ###
phi2 = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True, device_map=device)
phi2_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
phi2 = torch.compile(phi2, backend='eager') # optional