from concurrent.futures import ThreadPoolExecutor

import PIL.Image
from openai import OpenAI

from ..vlms import image_url
from .llava16 import llava16_vqa

def create_padded_crop(image: PIL.Image.Image, bounding_box):
    # Slightly expand the bounding box.
    x1 = bounding_box[0] - 10
    y1 = bounding_box[1] - 10
    x2 = bounding_box[2] + 10
    y2 = bounding_box[3] + 10

    # Crop the image to the bounding box.
    crop = image.crop((x1, y1, x2, y2))
    
    return crop

def describe_object(oai: OpenAI, image, bounding_box):
    visual_prompt = create_padded_crop(image, bounding_box)
    cmpl = oai.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "system",
                "content": "You accurately describe what is in an image that is presented to you. All statements must be supported by the image content."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url(visual_prompt)}},
                    {"type": "text", "text": "Please describe this object."}
                ]
            }
        ]
    )
    result = cmpl.choices[0].message.content
    return result

def describe_objects(image, bounding_boxes):
    oai = OpenAI()
    # Speed up this code with multithreading.
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda bounding_box: describe_object(oai, image, bounding_box), bounding_boxes)
    return list(results)

def describe_object_oss(image, bounding_box):
    item_image = create_padded_crop(image, bounding_box)
    result = llava16_vqa(item_image, "What is this object?") # type: ignore
    print("Result:", result)
    return result

def describe_objects_oss(image, bounding_boxes):
    # Speed up this code with multithreading.
    # with ThreadPoolExecutor() as executor:
    #     results = executor.map(lambda bounding_box: describe_object_oss(image, bounding_box), bounding_boxes)
    # return list(results)
    
    return list(
        map(lambda bounding_box: describe_object_oss(image, bounding_box), bounding_boxes)
    )

