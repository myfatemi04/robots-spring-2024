import copy
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from clients import llm_client, vlm_client
from detect_objects import Detection
from memory_bank_v2 import Memory, MemoryBank, MemoryKey, Retrieval
from memory_short_term import WorkingMemory
from object_detection_utils import draw_set_of_marks
from vlms import image_message


def parse_likelihood_response(response) -> Tuple[Optional[str], Dict[int, str]]:
    """
    FORMAT:
    ```
    Reasoning:
    The object is commonly used in this context.

    Choices:
    1: likely
    2: neutral
    3: unlikely
    ```
    """

    # Parse the response
    reasoning_match = re.search(r'Reasoning:(.*?)Choices:', response, re.DOTALL)
    choices_match = re.search(r'Choices:(.*)', response, re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    choices_str = choices_match.group(1).strip() if choices_match else None

    # Extract individual choices
    choices = {}
    if choices_str:
        for line in choices_str.split('\n'):
            if ':' in line:
                obj_num, choice = line.split(':')
                choices[int(obj_num.strip())] = choice.strip()

    return (reasoning, choices)

@dataclass
class ObjectSelectionPolicyState:
    task: str
    natural_language_plan: str
    code_plan: str
    object_type: str
    object_purpose: str
    image: PIL.Image.Image
    image_annotated: PIL.Image.Image
    detections: List[Detection]
    detections_recalled_memories: List[List[Retrieval]]

def format_object_detections(detections: List[Detection], descriptions: List[str], memories_per_object: List[List[Retrieval]]):
    prompt_string = 'Detections\n'
    for i, detection in enumerate(detections):
        center_x, center_y = detection.center
        
        prompt_string += f"Object ({i + 1})\nPixel location: ({center_x:.0f}, {center_y:.0f})\nDescription: {descriptions[i]}\n"
        for score, memory in memories_per_object[i]:
            prompt_string += f"Note: This object has a visual similarity score of {score:.2f} to something which you noted, \"{memory.value}\".\n"
        prompt_string += '\n'
    
    return prompt_string

def get_human_feedback_for_selection(base_rgb_image, selected_object_id, working_memory: WorkingMemory, detections_2d: List[Detection]):
    """
    Use human feedback to calculate rewards for each object
     * We should try to generalize as much as possible, so that less human feedback is necessary as time goes on
    """
    # now we can try to incorporate human feedback
    annots = draw_set_of_marks(base_rgb_image, detections_2d)
    print(f"Please look at the set of objects. Here, we have selected {selected_object_id+1}.")
    print(f"Is this acceptable? If not, is any other object acceptable instead?")
    plt.title("Available objects")
    plt.imshow(annots)
    plt.axis('off')
    plt.show()
    acceptable = 'y' == input("Is this acceptable? (y/n): ")
    if not acceptable:
        negative_object_id = selected_object_id
        selected_object_id = int(input("Which object should be selected instead? (1-indexed): ")) - 1
        if selected_object_id >= 0:
            # now we try to create a memory from this human feedback
            # however we gotta be real that most people will ignore this?
            reason_unacceptable = input("Why was the original choice unacceptable? ")

def get_selection_policy_gradient_estimate(state: ObjectSelectionPolicyState, fixed_memories: List[Tuple[int, Retrieval]], test_memories: List[Tuple[int, Retrieval]]):
    """
    Estimate the gradient of utility for this memory. We do this by simply removing each recalled memory and seeing how much
    the likelihoods change. Requires O(1 + (num test memories)) LLM inferences, which *could* be considered expensive?
    """
    original_memories = state.detections_recalled_memories
    original_reasoning, original_logits = get_selection_policy(state)
    
    n_objects = len(state.detections)
    assert len(fixed_memories) == n_objects and len(test_memories) == n_objects, "Mismatched number of memories compared to the number of object detections"

    delta_logits = []

    for held_out_memory_index in range(len(test_memories)):
        new_state = copy.copy(state)
        new_state.detections_recalled_memories = [[] for _ in range(n_objects)]

        for (object_id, fixed_memory) in fixed_memories:
            new_state.detections_recalled_memories[object_id].append(fixed_memory)

        for (memory_index, (object_id, test_memory)) in enumerate(test_memories):
            if memory_index == held_out_memory_index:
                continue

            new_state.detections_recalled_memories[object_id].append(test_memory)

        reasoning, logits = get_selection_policy(state)

        delta_logits.append(logits - original_logits)

    return delta_logits

def get_selection_policy(context: list):
    """
    Scope of this function:
     - Given an input state, output a policy for which objects to select
    """


    cmpl = vlm_client.chat.completions.create(
        model='gpt-4-vision-preview',
        messages=[*context]
    )
    reasoning, choices = parse_likelihood_response(cmpl.choices[0].message.content)

    # get logits
    logits = np.zeros(max(choices.keys()))
    for key, value in choices.items():
        if value.lower().strip() == 'unlikely':
            logits[key - 1] = -1
        elif value.lower().strip() == 'likely':
            logits[key - 1] = 1

    return (reasoning, logits)

'''
def __blah():
    object_scores = []

    if max(nretrievals_per_object) == 0:
        # we don't remember anything in particular about the objects in the scene
        object_scores = [0] * len(nretrievals_per_object)
    else:
        # have an LLM filtering step for these memories
        prompt = f"""{working_memory.serialize_current_plan()}

The following objects have been detected:
{description}

Rank the objects in the scene according to how likely they are to be the best choice.
Respond with 'likely', 'neutral', and 'unlikely' for each object. Format your list as follows:
```
1: likely
2: neutral
3: unlikely
```
""".strip()
        
        print("Prompt:")
        print(prompt)

        cmpl = llm_client.chat.completions.create(
            model='gpt-4-turbo',
            messages=[
                {"role": "system", "content": "You are a helpful human assistant who uses a robot API to help robots make motion plans."},
                {"role": "user", "content": prompt}
            ]
        )
        message_content = cmpl.choices[0].message.content
        print(message_content)
        object_scores = [int(x) for x in input("type the object scores inferred from above. 1 = likely, 0 = neutral, -1 = unlikely:").split(" ")]

        print("resulting object scores:", object_scores)

        # now we make a selection based on these scores
        # using a softmax-like approach maybe? we want to do some kind
        # of exploration i would think. this connects to RL if reward = following the human's
        # instructions accurately. maybe we can try to have the LLM quantify risk and use that
        # for exploration; but could leave that to future work.
        # - boltzmann exploration (with a low temperature)
        # - epsilon-greedy
        # also, sometimes when the human provides assumptions, it is possible that:
        # - the robot cannot physically execute the skill
        # - the robot fails to execute the skill
        # in cases where the human was wrong and the object was incorrectly used,
        # maybe we should have a way to record that.

        # i will use boltzmann exploration with a low temperature (maybe we can increase the temperature
        # as we get more data? would like to be able to automatically calibrate the temperature based on
        # confidence level though.)
        tau = 0.1
        object_scores = np.array(object_scores)
        object_scores = np.exp(object_scores / tau)
        object_scores /= object_scores.sum()
        # choose object
        object_id = np.random.choice(len(object_scores), p=object_scores)

        # now we can try to incorporate human feedback
        annots = draw_set_of_marks(base_rgb_image, detections_2d)
        print(f"Please look at the set of objects. Here, we have selected {object_id+1}.")
        print(f"Is this acceptable? If not, is any other object acceptable instead?")
        # print("However, we would like to know which objects would have been equivalently acceptable or should have been avoided.")
        plt.title("Available objects")
        plt.imshow(annots)
        plt.axis('off')
        plt.show()
        acceptable = 'y' == input("Is this acceptable? (y/n): ")
        if not acceptable:
            negative_object_id = object_id
            object_id = int(input("Which object should be selected instead? (1-indexed): ")) - 1
            if object_id >= 0:
                # now we try to create a memory from this human feedback
                # however we gotta be real that most people will ignore this?
                reason_unacceptable = input("Why was the original choice unacceptable? ")
                # from here we can try to infer some information about that object
                # it's also possible that it's not an intrinsic property of the object,
                # but rather something extrinsic
                prompt = f"""
We are trying to achieve the following task:
{working_memory.task}

You have written the following plan:
\"""
{working_memory.plan_natural_language}
\"""

We are in the middle of the code's execution. The line we are currently executing is
`scene.choose({object_type}, {purpose})`.

You originally selected object {object_id + 1} as the best choice. However, the selector has indicated that this object is unacceptable. They have provided the following reason:
{reason_unacceptable}

What relevant information should you remember about object {object_id + 1} in the future? Write a sentence or two.
Write without referring to "object {object_id + 1}" in particular; the object numbering might change in the future.
Imagine that this information will show up alongside the same object class in the future when it appears.
""".strip()
            print("Prompt:")
            print(prompt)

            pos_emb = detections_2d[object_id]['emb_augmented']
            neg_emb = detections_2d[negative_object_id]['emb_augmented']

            cmpl = llm_client.chat.completions.create(
                model='gpt-4-turbo',
                messages=[
                    {"role": "system", "content": "You are a helpful human assistant who uses a robot API to help robots make motion plans."},
                    {"role": "user", "content": prompt}
                ]
            )
            to_remember = cmpl.choices[0].message.content

            print("committing to short-term memory:", to_remember)
            print("positive ID:", object_id)
            print("negative ID:", negative_object_id)

            # now, we commit this to short-term memory
            self.lmp_planner.memory_bank.store(Memory(
                key=MemoryKey(
                    positive_examples=[*pos_emb],
                    negative_examples=[*neg_emb],
                ),
                value=to_remember,
            ))
'''
