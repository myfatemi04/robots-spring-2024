"""
Implements the main agent loop.

- Observe
- Retrieve?
- Act [Decide whether the robot needs to be used or not; formulate a response]. Format as interactive markdown.
- Reflect [Update the agent's memory bank with the new information].

What should be stored in working memory? / How should working memory be represented?
 - History should be represented as a sequence of dialogue turns
 - The past K interactions (including planning steps, etc.)
 - Retrieved text memories

Store as a sequence of events (e.g. Perception event, Action event [can be speech or code execution], Human event)
Then we can retrieve relevant information according to Perception and prior interactions.

Test case: coffee preferences

John asks for some coffee.
The robot grabs a coffee packet.
John realizes it's caffeinated, but he wants decaf.
The robot registers this preference in its memory bank (e.g. "john wants decaf" and "this coffee is caffeinated").

OK let's try to divide memory registration into 2 things:
(1) the object that was interacted with
(2) the objects that were not interacted with

We should log all object selections for feedback.
Using the policy gradient approach, we can try to estimate
the Q value of each object conditioned on the context (instructions)
where *memories* are the parameters of the policy.

Trying to formalize the agent loop. Maybe I should use the Soar architecture.

Memories should be retrieved at any stage where actions are selected. This *includes* the code as policies. But I
think for now we can just do it at the object selection stage.

Information about which objects should be selected may span several steps.

"""

import os
import pickle
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import matplotlib.pyplot as plt
import PIL.Image
from detect_objects import Detection, detect
from select_object_v2 import draw_set_of_marks
from lmp_planner import (StatefulLanguageModelProgramExecutor,
                         reason_and_generate_code)
from lmp_scene_api import Scene
from openai import OpenAI
from vlms import image_message


class EventType(Enum):
    VISUAL_PERCEPTION = 0
    INNER_MONOLOGUE = 1
    CODE_ACTION = 2
    VERBAL_FEEDBACK = 3

@dataclass
class Event:
    pass
    # type: EventType

@dataclass
class VisualPerceptionEvent(Event):
    scene: Scene

@dataclass
class CodeActionEvent(Event):
    rationale: str
    code: Optional[str]
    raw_content: str
    # Should store the values of anything that changes (e.g. freeze the state of the environment)

@dataclass
class ReflectionEvent(Event):
    # We reflect on an action.
    # For example, maybe we generate memories associated with certain objects.
    # Then we log the human feedback associated with those memories.
    # How do we refer to the objects though?
    reflection: str

@dataclass
class VerbalFeedbackEvent(Event):
    text: str
    variables: Optional[dict] = None

@dataclass
class ExceptionEvent(Event):
    exception_type: str
    text: str

@dataclass
class EventStream:
    events: List[Event] = field(default_factory=list)

    def write(self, event: Event):
        self.events.append(event)

with open("prompts/code_generation.md") as f:
    code_generation_prompt = f.read()

def create_primary_context(event_stream: EventStream):
    last_vision_event = [i for i, event in enumerate(event_stream.events) if isinstance(event, VisualPerceptionEvent)][-1]
    context = []
    for i, event in enumerate(event_stream.events):
        if isinstance(event, VisualPerceptionEvent):
            if i < last_vision_event:
                context.append({
                    'role': 'system',
                    'content': '<Prior observation>'
                })
            else:
                context.append({
                    'role': 'system',
                    'content': [
                        {'type': 'text', 'text': "Here is what you currently see."},
                        image_message(event.scene.imgs[0]), # type: ignore
                    ]
                })
        elif isinstance(event, ReflectionEvent):
            context.append({
                'role': 'assistant',
                'content': f"Reflection: {event.reflection}"
            })
        elif isinstance(event, VerbalFeedbackEvent):
            context.append({
                'role': 'user',
                'content': f'{event.text}',
            })
        elif isinstance(event, CodeActionEvent):
            context.append({
                'role': 'assistant',
                'content': event.raw_content,
            })
    context.append(
        {'role': 'system', 'content': code_generation_prompt}
    )
    return context

def create_vision_model_context(event_stream: EventStream):
    """
    We predict P(useful description | visual input and prior dialogue)
    """
    last_vision_event = [i for i, event in enumerate(event_stream.events) if isinstance(event, VisualPerceptionEvent)][-1]
    context = []
    for i, event in enumerate(event_stream.events):
        if isinstance(event, VisualPerceptionEvent):
            if i < last_vision_event:
                context.append({
                    'role': 'system',
                    'content': '<Prior observation>'
                })
            else:
                context.append({
                    'role': 'system',
                    'content': [
                        {'type': 'text', 'text': "Here is what you currently see."},
                        image_message(event.scene.imgs[0]), # type: ignore
                    ]
                })
        elif isinstance(event, ReflectionEvent):
            context.append({
                'role': 'assistant',
                'content': f"Reflection: {event.reflection}"
            })
        elif isinstance(event, VerbalFeedbackEvent):
            context.append({
                'role': 'user',
                'content': f'{event.text}',
            })
        elif isinstance(event, CodeActionEvent):
            context.append({
                'role': 'assistant',
                'content': event.raw_content,
            })
    context.append(
        {'role': 'system', 'content': code_generation_prompt}
    )
    return context

def agent_loop():
    event_stream = EventStream()

    def ask(prompt='[Robot called the ask() function without a prompt]:'):
        result = input(prompt)
        event_stream.write(VerbalFeedbackEvent(result))
        return result

    # rgbd = RGBD(num_cameras=1)
    code_executor = StatefulLanguageModelProgramExecutor(vars={"ask": ask})
    client = OpenAI()

    # start_i = 0
    # if os.path.exists("event_stream.pkl"):
    #     with open("event_stream.pkl", "rb") as f:
    #         event_stream = pickle.load(f)
    #     start_i = sum(1 for evt in event_stream.events if isinstance(evt, VisualPerceptionEvent))
    #     print(event_stream)

    # Create a custom event stream
    scene = Scene([PIL.Image.open("sample_images/IMG_8651.jpeg")])
    # detections = detect(scene.imgs[0], "deck of cards")
    # drawn = draw_set_of_marks(scene.imgs[0], detections)
    # plt.title('Detections')
    # plt.imshow(drawn)
    # plt.axis('off')
    # plt.show()
    
    # event_stream.write(VisualPerceptionEvent(scene))
    event_stream.write(VerbalFeedbackEvent("Please put these items away"))
    
    for i in range(1):
        # rgbs, pcds = rgbd.capture()
        # imgs = [PIL.Image.fromarray(rgb) for rgb in rgbs]
        imgs = [PIL.Image.open("sample_images/IMG_8651.jpeg")]
        scene = Scene(imgs)
        event_stream.write(VisualPerceptionEvent(scene))

#         rationale = """
# Reasoning:
# The task is to provide a snack from the available options in the image, which include a bag of Blue Diamond Almonds, a packet of Teddy Grahams, and a bag of Seasoned Croutons. The almonds and Teddy Grahams are typical snack choices, while croutons are generally used in salads. Given that the teddy grahams and almonds are more conventional snack options, I would choose one of these.

# Short Plan:
# 1. Decide which snack to pick up; considering typical snack preferences, I will choose the more universally appealing snack, which are the almonds or Teddy Grahams.
# 2. Locate the chosen snack pack using the scene detection.
# 3. Direct the robot to move its hand to the location of the snack pack.
# 4. Grasp the snack pack.
# 5. Move the robot hand back to the starting position with the snack pack.

# Code Implementation:
# <Code block>"""
#         code = """
# # Assume the Teddy Grahams are the chosen snack as they are likely more appealing to a larger audience, including children
# snack = scene.choose('packet', 'Teddy Grahams')

# # Direct the robot to move its hand to the location of the chosen snack pack
# robot.move_to(snack.centroid)

# # Grasp the snack pack
# robot.grasp(snack)

# # Move the robot hand back to a predetermined starting position with the snack pack
# # Assuming the starting position is [0, 0, 0] for this example
# robot.move_to([0, 0, 0])
# """
#         raw_content = rationale.replace("<Code block>", code)

        context = create_primary_context(event_stream)
        rationale, code, raw_content = reason_and_generate_code(context, imgs[0], client)
        event_stream.write(CodeActionEvent(rationale, code, raw_content))

        print("Reasoning and code generation:")
        print(raw_content)

        if code is not None:
            code_executor.update(scene=scene)
            status, output = code_executor.execute(code)
            if not status:
                assert output is not None, "Cannot return a False status without an output."
                print(f"Error: {output}")
                event_stream.write(ExceptionEvent("CodeExecutionError", output))

        # if i == 0:
        #     with open("event_stream.pkl", "wb") as f:
        #         pickle.dump(event_stream, f)

        # Use an ask/tell API to interact with the human.
        # All interactions will be performed with code, including speaking to the human.
        # LLM requests access to the robot. (e.g. robot = access_robot())

    pass

if __name__ == '__main__':
    agent_loop()
