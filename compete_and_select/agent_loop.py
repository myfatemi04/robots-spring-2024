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

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import PIL.Image
from lmp_planner import StatefulLanguageModelProgramExecutor, reason_and_generate_code
from lmp_scene_api import Scene
from openai import OpenAI
from rgbd import RGBD
from vlms import image_url


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

class EventStream:
    def __init__(self):
        self.events = []

    def add_event(self, event: Event):
        self.events.append(event)

    def get_last_event(self):
        return self.events[-1]

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
                        {'type': 'image_url', 'image_url': image_url(event.scene.imgs[0])}
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
    return context

def agent_loop():
    event_stream = EventStream()
    # rgbd = RGBD(num_cameras=1)
    code_executor = StatefulLanguageModelProgramExecutor()
    client = OpenAI()
    for _ in range(10):
        # rgbs, pcds = rgbd.capture()
        # imgs = [PIL.Image.fromarray(rgb) for rgb in rgbs]
        imgs = [PIL.Image.open("sample_images/IMG_8650.jpeg")]
        pcds = [None]
        scene = Scene(imgs, pcds, [f'img{i}' for i in range(len(imgs))])
        event_stream.add_event(VisualPerceptionEvent(scene))

        context = create_primary_context(event_stream)
        rationale, code, raw_content = reason_and_generate_code(context, imgs[0], client)

        print("Reasoning and code generation:")
        print(raw_content)

        if code is not None:
            code_executor.execute(code)

        event_stream.add_event(CodeActionEvent(rationale, code, raw_content))

        # Use an ask/tell API to interact with the human.
        # All interactions will be performed with code, including speaking to the human.
        # LLM requests access to the robot. (e.g. robot = access_robot())

    pass

if __name__ == '__main__':
    agent_loop()
