import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import transformers
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LLaVA model.
model: transformers.LlavaForConditionalGeneration = transformers.LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf") # type: ignore
model = model.to(device=device)
processor = transformers.AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf") # type: ignore


prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"

image = Image.open("IMG_8400.png")
image_np = np.array(image)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device=device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# Visualize the image
plt.imshow(image_np)
plt.show()

print("Description generated by Llava:")
print(result)

# make this point to the first demo
image = images[0]
image_np = image.cpu().permute(1, 2, 0).numpy()

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device=device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# Visualize the image
plt.imshow(image_np)
plt.show()

print("Description generated by Llava:")
print(result)

image = images[0]
image_np = image.cpu().permute(1, 2, 0).numpy()

inputs = processor(text="<image>\nUSER: There is a drawer here. What direction is it facing?\nASSISTANT:", images=image, return_tensors="pt").to(device=device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# Visualize the image
plt.imshow(image_np)
plt.show()

print("Description generated by Llava:")
print(result)

image = demos[0][0].overhead_rgb
image_np = image

inputs = processor(text="<image>\nUSER: There is a drawer here. What direction is it facing?\nASSISTANT:", images=image, return_tensors="pt").to(device=device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# Visualize the image
plt.imshow(image_np)
plt.show()

print("Description generated by Llava:")
print(result)
