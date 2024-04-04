from generate_object_candidates import draw_set_of_marks, detect
from vlms import gpt4v_plusplus as gpt4v
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Context:
    # useful if we want to have *really* long horizon.
    # think of these as perception->reasoning->acting->outcome tuples
    # from a significantly earlier episode.
    retrieved_examples: list
    # stores a list of [image, string] alternating
    # that is, perception -> reasoning -> acting triplets
    history: list

# defines an openai function to make a keypoint selection
select_grasp_location_function = {
    "type": "function",
    "function": {
        "description": "Controls where the robot arm grabs.",
        "name": "select_grasp_location",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "object_id": {"type": "string"},
                "relative_x_percent": {"type": "integer"},
                "relative_y_percent": {"type": "integer"},
            },
            "required": ["reasoning", "object_id", "relative_x_percent", "relative_y_percent"]
        }
    }
}

"""
# `context` includes the past trajectory we've seen so far.
`reasoning` and `plan_str` contain information to inform this
function of *why* we're looking for `target_object_label`. This
will enable the VLM to select the correct object afterwards.
"""
def select_next_action(image, instructions, reasoning, plan_str, target_object_label):
    image = image.convert("RGB")
    dets = detect(image, target_object_label)

    if len(dets) == 0:
        return (False, "no_object_detections")

    # Draw set-of-marks prompts to select the object.
    set_of_marks_prompt = draw_set_of_marks(image, dets)

    plt.title("Set of marks prompt")
    plt.imshow(set_of_marks_prompt)
    plt.axis('off')
    plt.show()

    # Select object
    result = gpt4v([(set_of_marks_prompt, f"""
## Instructions
{instructions}

## Observations
{reasoning}

## Plan
{plan_str}

## Action

Look at the label associated with the correct object to interact with. Think carefully about how to interact with the object, considering what specific part
of the object to interact with. Then, write the relative position within that object's bounding
box to grab. Assume positive x is to the right, and positive y is upward. Write your answer as
a percentage. For example, if you think the robot should grab the top of object #5, one possible
control command is "Object: #5, Position: (50, 90)". This represents selecting the top of object
#5, the horizonal center of the bounding box, and the top of the bounding box. Please write your
percentages with high precision. Object IDs are written directly ABOVE the bounding box they correspond to.
""".strip())],
        max_tokens=384,
        tools=[select_grasp_location_function],
        tool_choice={
            "type": "function",
            "function": {"name": "select_grasp_location"}
        }
    )

    return (True, {
        "action_selection": result,
        "detections": dets
    })
