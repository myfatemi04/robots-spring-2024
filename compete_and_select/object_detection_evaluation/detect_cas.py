from transformers import AutoModelForCausalLM, AutoTokenizer
import PIL.Image
import torch
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import numpy as np
import torchvision.ops as ops
from transformers import Owlv2ForObjectDetection, Owlv2Processor

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

### Object Detection Model ###
model_name = 'google/owlv2-large-patch14-ensemble'
processor: Owlv2Processor = Owlv2Processor.from_pretrained(model_name) # type: ignore
model = torch.compile(Owlv2ForObjectDetection.from_pretrained(model_name, device_map=device), backend='eager') # type: ignore

### VLM ###
model_type = 'multimodal_phi'

if model_type == 'moondream':
    from moondream_model import vlm, tokenizer

    # optional compilation
    # vlm.vision_encoder = torch.compile(vlm.vision_encoder, backend='eager')
    # vlm.text_model = torch.compile(vlm.text_model, backend='eager')
elif model_type == 'prismatic':
    from prismatic_model import vlm
elif model_type == 'multimodal_phi':
    from multimodal_phi_model import vlm, processor as vlm_processor

from phi2_model import phi2, phi2_tokenizer

# Modified version of the one in
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlv2/image_processing_owlv2.py
# to output logits for all labels (in case of overlapping or etc.)
# rather than just the max label.
def post_process_object_detection(
    outputs, threshold: float = 0.1, target_sizes = None
):
    from transformers.image_transforms import center_to_corners_format
    
    objectness_logits, logits, boxes = outputs.objectness_logits, outputs.logits, outputs.pred_boxes

    if target_sizes is not None:
        if len(logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

    scores = torch.sigmoid(logits)
    # `keep` has shape [batch size, token count]
    keep = torch.any(scores > threshold, dim=-1)

    # Convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(boxes)

    # Convert from relative [0, 1] to absolute [0, height] coordinates
    if target_sizes is not None:
        if isinstance(target_sizes, List):
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
        else:
            img_h, img_w = target_sizes.unbind(1)

        # Rescale coordinates, image is padded to square for inference,
        # that is why we need to scale boxes to the max size
        size = torch.max(img_h, img_w)
        scale_fct = torch.stack([size, size, size, size], dim=1).to(boxes.device)

        boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b, o, k in zip(scores, logits, boxes, objectness_logits, keep):
        score = s[k]
        logit = l[k]
        box = b[k]
        objectness_logit = o[k]
        results.append({"scores": score, "logits": logit, "boxes": box, "objectness_logits": objectness_logit})

    return results

def _detect_coarse(image, labels, threshold=0.1):
    with torch.no_grad():
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
    results = results[0]
    boxes, scores, logits, objectness_logits = results["boxes"], results["scores"], results["logits"], results["objectness_logits"]
    
    # Filter by non-maximum suppression and box size
    keep = ops.nms(boxes, objectness_logits, iou_threshold=0.2)
    
    box_sizes = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) / (image.height * image.width)
    
    keep = keep[box_sizes[keep] < 0.25]

    return (boxes[keep], scores[keep], logits[keep], outputs)

# A shared OwlV2 wrapper to save memory.
class OwlV2Wrapper:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
        
    def __call__(self, image, captions, threshold=0.1):
        detections = []
        boxes, scores, logits, outputs = _detect_coarse(image, captions, threshold)
        for i in range(len(boxes)):
            detections.append({"box": boxes[i], "xyxy": boxes[i].detach().cpu().numpy(), "scores": scores[i], "logits": logits[i]})
        return detections
    
owlv2 = OwlV2Wrapper(processor, model)

# Generate a short prompt.
make_prompt = lambda caption, target: f"Question: An object is described as {caption}. Could this be {target}? Answer:"
yes_token, no_token = phi2_tokenizer([" Yes", " No"], return_tensors='pt').to(device).input_ids[:, 0]

def get_caption_logits(caption, target):
    caption = caption.lower()
    target = target.lower()
    if caption.endswith("."):
        caption = caption[:-1]
    if target.endswith("."):
        target = target[:-1]
    if caption.startswith("the "):
        caption = "a " + caption[4:]
    if target.startswith("the "):
        target = "a " + target[4:]
    prompt = make_prompt(caption, target)
    # print(prompt)
    model_input = phi2_tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        pred = phi2(**model_input)
        
    yn = pred.logits[0, -1, [yes_token, no_token]]
    y, n = yn - yn.min()
    
    return (y.item(), n.item())

def get_caption_logit(caption, target):
    y, n = get_caption_logits(caption, target)
    return y - n

def generate_caption_phi3(image, bbox):
    crop = image.crop(tuple(int(x) for x in bbox))
    
    # Pad to create a square image
    max_dim = max(crop.width, crop.height)
    crop_pad = PIL.Image.new('RGB', (max_dim, max_dim))
    crop_pad.paste(crop, ((max_dim - crop.width) // 2, (max_dim - crop.height) // 2))
    crop = crop_pad
    
    prompt = "<|user|>\n<|image_1|>Please describe this image in terms of physical attributes.<|end|>\n<|assistant|>\n"
    
    with torch.no_grad():
        inputs = vlm_processor(prompt, [crop_pad], return_tensors="pt").to("cuda:0")

        generation_args = {
            "do_sample": False,
            "max_new_tokens": 64,
        }

        generate_ids = vlm.generate(**inputs, eos_token_id=vlm_processor.tokenizer.eos_token_id, **generation_args) 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = vlm_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

def detect(image: PIL.Image.Image, targets: list, coarse_threshold_=0.1, verbose=False):
    import matplotlib.pyplot as plt
    # (1) Detect objects that should be vaguely similar to `targets` with OwlV2.
    #  - Keep the logits for each class so we can calculate precision and recall more easily.
    # (2) Generate captions for each bounding box.
    # (3) Collect VLM logits for (yes | no) for each item. Use phi-2 for this.
    
    import time
    
    t0 = time.time()
    
    coarse_detection_outputs = (boxes, scores, logits, raw_outputs) = _detect_coarse(image, targets, threshold=coarse_threshold_)
    
    if verbose:
        print("Generated coarse detections.")
    
    t1 = time.time()

    # Caption each of the boxes with the VLM.
    crops = [image.crop(tuple(int(x) for x in box)) for box in boxes]
    
    # Try batched inference.
    if model_type == 'moondream':
        captions = []
        with torch.no_grad():
            i = 0
            while i < len(crops):
                crops_ = crops[i:i + 4]
                i += 4
                captions.extend(vlm.batch_answer(crops_, ["Briefly describe this image in terms of objective, physical attributes."] * len(crops_), tokenizer))
    elif model_type == 'prismatic':
        do_batched = False
        if do_batched:
            # Batched inference.
            image_transform = vlm.vision_backbone.image_transform
            pixel_values = torch.stack([image_transform(crop) for crop in crops], dim=0)
            print(pixel_values.shape)
            prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Please describe this image based on physical attributes. ASSISTANT:"
            captions = vlm.generate_batch(
                pixel_values,
                [prompt] * len(crops),
                do_sample=False,
                max_new_tokens=64,
                min_length=1
            )
        else:
            captions = []
            # Serial generation.
            for crop in crops:
                print(f"Generating caption {len(captions) + 1} / {len(crops)} ...")
                with torch.no_grad():
                    # Pad to create a square image
                    max_dim = max(crop.width, crop.height)
                    crop_pad = PIL.Image.new('RGB', (max_dim, max_dim))
                    crop_pad.paste(crop, ((max_dim - crop.width) // 2, (max_dim - crop.height) // 2))
                    crop = crop_pad

                    caption = vlm.generate(
                        crop,
                        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Please describe this image based on physical attributes. ASSISTANT:",
                        # do_sample=True,
                        # temperature=0.4,
                        do_sample=False,
                        max_new_tokens=64,
                        min_length=1,)
                    captions.append(caption)
                    # plt.title(caption)
                    # plt.imshow(crop)
                    # plt.show()
    elif model_type == 'multimodal_phi':
        captions = []
        # Serial generation.
        for crop in crops:
            print(f"Generating caption {len(captions) + 1} / {len(crops)} ...")
            with torch.no_grad():
                # Pad to create a square image
                max_dim = max(crop.width, crop.height)
                crop_pad = PIL.Image.new('RGB', (max_dim, max_dim))
                crop_pad.paste(crop, ((max_dim - crop.width) // 2, (max_dim - crop.height) // 2))
                
                prompt = "<|user|>\n<|image_1|>Please describe this image in terms of physical attributes.<|end|>\n<|assistant|>\n"
                
                inputs = processor(prompt, [crop_pad], return_tensors="pt").to("cuda:0")
                
                generation_args = {
                    "temperature": 0.0,
                    "do_sample": False,
                    "max_new_tokens": 500,
                }
                
                generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

                # remove input tokens 
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                captions.append(response)
    
    if verbose:
        print("Generated captions.")
    
    t2 = time.time()
    
    # Now, get LLM logit results.
    binary_logits = torch.zeros((len(captions), len(targets)))
    for i, caption in enumerate(captions):
        for j, target in enumerate(targets):
            # We could do some caching for this. But the memory footprint might be high.
            y, n = get_caption_logits(caption, target)
            # categorical logits => binary logit
            # exp(y)/(exp(y)+exp(n)) = 1/(1 + exp(-(y - n))
            binary_logits[i, j] = y - n
            
    if verbose:
        print("Generated logits.")
            
    t3 = time.time()
    
    if verbose:
        print(f"OwlV2 detection time: {t1 - t0:.2f}")
        print(f"Caption generation time: {t2 - t1:.2f}")
        print(f"Logit generation time: {t3 - t2:.2f}")
    
    return (captions, crops, binary_logits, coarse_detection_outputs)
