import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from vlms import scale_image, gpt4v, llava


def parse_plan(plan):
    # Extract the first bulleted list.
    lines = plan.strip().split("\n")
    steps = []
    for line in lines:
        # $ = end of line
        pattern = re.compile(fr"(\d+)\. Reference point: (.+?), Action: (.+?)$")
        match = pattern.match(line)
        if match is None:
            if len(steps) == 0:
                continue
            else:
                break
            
        # counter = int(match.group(1))
        object_ = match.group(2)
        action = match.group(3)
        
        steps.append((action, object_))
        
    return steps

def block(title, s):
    print((">" * 10), title)
    print(s)
    print(("<" * 10), title)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_action_candidates(image, task, vlm='gpt4v'):
    # plt.title("Input Image.")
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    
    # Simple prompt engineering.
    if vlm == 'llava':
        prompt = f"""
USER: {task}. How could I start doing this? Give me a very concrete / literal set of steps.
First, write "Observations:" followed by some of your observations. Then, write "Plan:", followed by a very specific set of objects
and actions to manipulate them.
ASSISTANT: Observations:"""
        
        reasoning = "Observations: " + llava(image, prompt)
    
        block("Reasoning Generated By LLaVA", reasoning)
        
        prompt_2 = f"""{prompt}{reasoning}
        USER: Now, give instructions to a robot to follow these steps. Write your answers in the format "Object: [object], Action: [action]".
        ASSISTANT: 1. Object:"""
        plan_str = "1. Object: " + llava(image, prompt_2).strip()
        
        block("Parsed Plan Section", plan_str)
    else:
        prompt = f"""{task}. Given this image, how could I start doing this?e
Give me a very concrete / literal set of steps, as if you were instructing a robot
with very little semantic understanding, who can only use visual or physical descriptions
of how to interact with objects.

Begin your response with the exact string "Observations:" followed by some of your observations. Use this time to understand
the layout of the scene and what you could do to complete the task at hand.

Then, write "Plan:", followed by very literal text descriptions for what the robot's arm should do. You will do this in two steps: The first, selecting a reference point,
and the second, a short description for the arm's position relative to that reference point. Write your answers in the format "Reference point: ..., Action: ...". For example,
if you want to open a door, write "Reference point: Door handle, Action: Twist". For "reference point", only write the name of the object being referred to, and potentially an
adjective. Do not describe the reference point in relation to other objects.

Begin your answer with the string "1. ", and continue with a numbered list of the actions to take. Notes:
 - "Reference point" should be a text description of what object the robot should interact with in the scene. It should NEVER refer to the robot arm itself.
   This command will be *sent* to the robotic arm, and the robotic arm will use this to identify an object to interact with. Do not use propositions or spatial adjectives.
 - "Action" should refer to the action the robot arm should take with the object.
 - Both "Object" and "Action" should be very brief descriptions (i.e. 3-5 words).
 - Ensure you adhere to the format "Reference point: [object], Action: [action]". The two text descriptions must be on the same line, separated by a comma.
""".strip()
        
        response = gpt4v(image, prompt, temperature=0)
#         response = """
# Observations: There is a robot arm located in a room with a desk and various objects. On a dark circular table in front of the robot, there is a red block and two green cups placed upside-down, one of which has a blue cloth partially underneath it. Also observable are other miscellaneous objects, like tape and batteries, but these are irrelevant to the task. The red block is positioned to the robot's right and slightly ahead of it, while the cups are to the left. We assume the goal is to have the robot place the block into one of the green cups.

# Plan:
# 1. Object: Red block, Action: Grip the block.
# 2. Object: Red block, Action: Lift the block up.
# 3. Object: Green cup, Action: Move the block over the cup.
# 4. Object: Green cup, Action: Lower the block into the cup.
# 5. Object: Red block, Action: Release grip on the block.
#         """.strip()
        
        block("Raw GPT-4V Response", response)
        
        plan_start_index = response.index("Plan:")
        reasoning = response[:plan_start_index]
        
        block("Parsed Observations Section", reasoning)
        
        plan_str = response[plan_start_index + 5:].strip()
        
        block("Parsed Plan Section", plan_str)
    
    plan = parse_plan(plan_str)
    
    if len(plan) == 0:
        print("ERROR: Plan parsing failed.")
        return
    
    block("Parsed Action Literals", '\n'.join(str(x) for x in plan))

    return (reasoning, plan_str, plan)
    
if __name__ == '__main__':
    if False:
        parse_plan("""
    1. Object: Laptop, Action: Close
    2. Object: Laptop, Action: Put in backpack
    3. Object: Backpack, Action: Remove from floor
    4. Object: Backpack, Action: Place on chair
    5. Object: Laptop, Action: Open
    6. Object: Laptop, Action: Put in backpack
    7. Object: Backpack, Action: Remove from chair
    8. Object: Backpack, Action: Place on floor
    9. Object: Laptop, Action: Close
    10. Object: Laptop, Action: Put in backpack
    """)

    image = scale_image(Image.open("droid_sample.png").convert("RGB"))
    generate_action_candidates(image, "Pour some oil in the pan", vlm='gpt4v')