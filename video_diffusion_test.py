import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import PIL.Image

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()
pipe.unet.enable_forward_chunking()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
with torch.no_grad():
    frames = pipe(image, decode_chunk_size=1, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1, num_frames=6).frames[0]

for i, frame in enumerate(frames):
    frame.save(f"generated_{i}.png")

export_to_video(frames, "generated.mp4", fps=7)
