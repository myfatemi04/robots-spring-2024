from typing import List, Tuple, Optional, Dict
from .object_detection_utils import draw_set_of_marks
from .describe_objects import describe_objects
from .vlms import image_url
from openai import OpenAI
import numpy as np
import os

def parse_likelihood_response(response: str) -> Tuple[str, Dict[int, str]]:
    """
    FORMAT:
    ```
    Reasoning:
    The object is commonly used in this context.

    Choices:
    1: likely (... and potential explanation after this point)
    2: neutral (...)
    3: unlikely (...)
    ```
    """

    # Parse the response
    choices_index = response.find('Choices:')
    reasoning = response[:choices_index].strip()
    choices_str = response[choices_index + 8:].strip()

    # Extract individual choices
    choices = {}
    for line in choices_str.strip().split('\n'):
        if ':' in line:
            obj_num, choice = line.split(':')
            try:
                choices[int(obj_num.strip())] = choice.strip()
            except:
                print("WARN: Skipping line:", line, "in choices_str:")
                print(choices_str)
                pass
        else:
            # break at the first line without an object.
            break
            
    if len(choices) == 0:
        print("No choices found:")
        print(response)

    return (reasoning, choices)

def format_object_detections(bboxes: List[Tuple[int, int, int, int]], descriptions: Optional[List[str]]): # , memories_per_object: List[List[Retrieval]]):
    prompt_string = 'Detections\n'
    for i, bbox in enumerate(bboxes):
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        prompt_string += f"Object ({i + 1})\nPixel location: ({center_x:.0f}, {center_y:.0f})\n"
        if descriptions is not None:
            prompt_string += f"Description: {descriptions[i]}\n"
            
        # for score, memory in memories_per_object[i]:
        #     prompt_string += f"Note: This object has a visual similarity score of {score:.2f} to something which you noted, \"{memory.value}\".\n"
            
        prompt_string += '\n'
    
    return prompt_string

def get_selection_policy(context: list):
    """
    Scope of this function:
     - Given an input state, output a policy for which objects to select
    """
    
    vlm_client = OpenAI()
    
    if not os.path.exists("lm_request_cache"):
        os.mkdir("lm_request_cache")
    hsh = hash(str(context))
    if os.path.exists(f"lm_request_cache/{hsh}.txt"):
        with open(f"lm_request_cache/{hsh}.txt") as f:
            response = f.read()
            print("[cache hit]")
    else:
        response = vlm_client.chat.completions.create(
            model='gpt-4-vision-preview',
            messages=[*context]
        ).choices[0].message.content
        with open(f"lm_request_cache/{hsh}.txt", "w") as f:
            f.write(response)
    assert response is not None

    print("Created object selection policy:")
    print(response)

    reasoning, choices = parse_likelihood_response(response)

    # get logits
    logits = np.zeros(max(choices.keys()))
    for key, value in choices.items():
        if value.lower().strip().startswith('unlikely'):
            logits[key - 1] = -1
        elif value.lower().strip().startswith('likely'):
            logits[key - 1] = 1
        elif value.lower().strip().startswith('neutral'):
            logits[key - 1] = 0
        else:
            print("Warning: unrecognized choice", value)

    return (reasoning, logits, response)

import threading

SOM_lock = threading.Lock()
SOM_counter = 0

def select_with_vlm(image, bounding_boxes, target_object, descriptions, dry_run):
    global SOM_counter
    
    # given a set of detections, determine which is the most likely.
    # (1) describes the objects with a VLM
    # (2) puts the resulting objects into a text description
    
    object_detections_string = format_object_detections(bounding_boxes, descriptions) # , retrievals)
    
    # used to prevent multiple threads from editing the matplotlib figure
    SOM_lock.acquire()
    annotated_image = draw_set_of_marks(image, bounding_boxes)
    annotated_image.save(f"plots/set_of_marks__{SOM_counter:03d}.png")
    SOM_counter += 1
    SOM_lock.release()
    
    if dry_run:
        return {"reasoning": None, "logits": None, "response": None}

    context = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Here is what you currently see."}, {"type": "image_url", "image_url": {"url": image_url(annotated_image)}}]
        },
        {
            "role": "user",
            "content": f"""It is now time for you to select an object to interact with. Target object: {target_object}
            
The following objects have been detected:
{object_detections_string}

Rank the objects in the scene according to how likely they are to be the best choice.
Respond with 'likely', 'neutral', and 'unlikely' for each object. Format your response as follows:
```
Reasoning:
(Your reasoning goes here)

Choices:
1: likely
2: neutral
3: unlikely

Answer as if you are controlling a hypothetical robot. Assume that object detections have been
filtered to be in the reachable zone of the robot.
"""
        }
    ]
    
    reasoning, logits, response = get_selection_policy(context)
    
    return {
        "reasoning": reasoning,
        "logits": logits,
        "response": response,
    }        
    
