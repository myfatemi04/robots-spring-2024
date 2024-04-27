import base64
import io
import json
import os

import hjson
import PIL.Image as Image
from openai import OpenAI

llava_model = None
llava_processor = None
def load_llava():
    global llava_model, llava_processor
    
    from transformers import AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration# , LlavaLlamaForCausalLM
    
    model_id = "llava-hf/llava-1.5-13b-hf"
    llava_model = LlavaForConditionalGeneration.from_pretrained(model_id).to('cuda')
    # bf16. is this small enough?
    # model_id = "liuhaotian/llava-v1.6-34b"
    # llava_model = AutoModelForCausalLM.from_pretrained(model_id).to('cuda')

    llava_processor = AutoProcessor.from_pretrained(model_id)

if os.path.exists("key"):
    os.environ['OPENAI_API_KEY'] = open("key").read()
else:
    print("[WARN] No OpenAI API key file found.")

def llava(image, text, max_new_tokens=384):
    global llava_model, llava_processor
    
    if llava_model is None:
        load_llava()

    prompt = f"<image>{text}"
    inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to('cuda')
    
    generate_ids = llava_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = llava_processor.batch_decode(
        generate_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return generated

def pil_image_to_base64(image):
    """
    Convert a PIL Image to base64 format.

    Args:
    - image (PIL Image): Input image.

    Returns:
    - str: Base64 representation of the image.
    """
    # Convert the PIL Image to bytes
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()

    # Convert the bytes to base64
    base64_image = base64.b64encode(img_byte_array).decode('utf-8')
    return base64_image

def scale_image(image, max_dim=1024):
    """
    Resize the image so that the maximum dimension is `max_dim`.
    
    Args:
    - image (PIL Image): Input image
    - max_dim (int): Maximum dimension (width or height) of the output image
    
    Returns:
    - PIL Image: Resized image
    """
    width, height = image.size
    if max(width, height) > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(max_dim * height / width)
        else:
            new_height = max_dim
            new_width = int(max_dim * width / height)
        image = image.resize((new_width, new_height), Image.BICUBIC)
    return image

def image_message(image: Image.Image):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{pil_image_to_base64(image)}"
        }
    }

def gpt4v(image, text, max_new_tokens=384, **kwargs):
    client = OpenAI()
    
    response = client.chat.completions.create(
      model="gpt-4-vision-preview",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": text},
            image_message(image),
          ],
        }
      ],
      max_tokens=max_new_tokens,
      **kwargs,
    )
    
    return response.choices[0].message.content

def gpt4v_plusplus(msgs, **kwargs):
    client = OpenAI()

    messages = []
    role = "user"
    for msg in msgs:
        if type(msg) == str:
            messages.append({"role": role, "content": msg})
        elif type(msg) == tuple:
            messages.append({"role": role, "content": [
                x if type(x) == str else image_message(x) for x in msg
            ]})
        elif msg is None:
            pass
        else:
            assert False, "Unknown message type: " + str(type(msg))
        role = "user" if role == "assistant" else "assistant"

    print(":::: Calling GPT-4V ::::")
    # print(messages)
    
    response = client.chat.completions.create(
      model="gpt-4-vision-preview",
      messages=messages,
      **kwargs,
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls is not None and len(tool_calls) > 0:
        # for my purposes I will always know if a function was called
        try:
            return json.loads(tool_calls[0].function.arguments)
        except:
            try:
                return hjson.loads(tool_calls[0].function.arguments)
            except:
                print("could not parse with python json OR hjson. why can't openai just use YAML or some state machine to constrain outputs??")
                print(tool_calls[0].function.arguments)
    else:
        return response.choices[0].message.content
