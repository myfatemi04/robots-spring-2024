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

import cv2
import matplotlib.pyplot as plt
import PIL.Image
from agent_state import AgentState
from event_stream import (CodeActionEvent, EventStream, ExceptionEvent,
                          ReflectionEvent, VerbalFeedbackEvent,
                          VisualPerceptionEvent)
from lmp_planner import (StatefulLanguageModelProgramExecutor,
                         reason_and_generate_code)
from lmp_scene_api import Scene
from memory_bank_v2 import MemoryBank
from openai import OpenAI
from select_object_v2 import draw_set_of_marks
from vlms import image_message

with open("prompts/code_generation.md") as f:
    code_generation_prompt = f.read()

def create_primary_context(event_stream: EventStream, image_observation_overwrite=None):
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
                        image_message(image_observation_overwrite or event.imgs[0]), # type: ignore
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
    from rgbd import RGBD

    event_stream = EventStream()
    memory_bank = MemoryBank()
    agent_state = AgentState(event_stream, memory_bank)

    def ask(prompt='[Robot called the ask() function without a prompt]:'):
        result = input(prompt)
        event_stream.write(VerbalFeedbackEvent(result))
        return result

    rgbd = RGBD(num_cameras=1)
    code_executor = StatefulLanguageModelProgramExecutor(vars={"ask": ask})
    client = OpenAI()


    # Wait for calibration
    has_pcd = False
    while not has_pcd:
        (rgbs, pcds) = rgbd.capture()
        has_pcd = pcds[0] is not None
        plt.title("Camera 0")
        plt.imshow(rgbs[0])
        plt.pause(0.05)

    print(rgbs[0].shape)

    # start_i = 0
    # if os.path.exists("event_stream.pkl"):
    #     with open("event_stream.pkl", "rb") as f:
    #         event_stream = pickle.load(f)
    #     start_i = sum(1 for evt in event_stream.events if isinstance(evt, VisualPerceptionEvent))
    #     print(event_stream)

    # Create a custom event stream
    scene = Scene([PIL.Image.fromarray(rgbs[0])], None, agent_state)
    # scene = Scene([PIL.Image.open("sample_images/IMG_8651.jpeg")], None, agent_state)
    # detections = detect(scene.imgs[0], "deck of cards")
    # drawn = draw_set_of_marks(scene.imgs[0], detections)
    # plt.title('Detections')
    # plt.imshow(drawn)
    # plt.axis('off')
    # plt.show()

    scene.imgs[0].save("sample_images/oculus_and_headphones.png")

    return
    
    # event_stream.write(VisualPerceptionEvent(scene))
    event_stream.write(VerbalFeedbackEvent("Please pick up the Oculus controller."))
    
    for i in range(2):
        # rgbs, pcds = rgbd.capture()
        # imgs = [PIL.Image.fromarray(rgb) for rgb in rgbs]
        # imgs = [PIL.Image.open("sample_images/IMG_8651.jpeg")]
        # scene = Scene(imgs, None, agent_state)
        # event_stream.write(VisualPerceptionEvent(imgs, [None] * len(imgs)))
        event_stream.write(VisualPerceptionEvent(scene.imgs, [None]))

        context = create_primary_context(event_stream)
        rationale, code, raw_content = reason_and_generate_code(context, client)
        event_stream.write(CodeActionEvent(rationale, code, raw_content))

        print("Reasoning and code generation:")
        print(raw_content)

        if code is not None:
            code_executor.update(scene=scene)
            status, output = code_executor.execute(code)
            if not status:
                print(f"Error: {output}")
                event_stream.write(ExceptionEvent("CodeExecutionError", output)) # type: ignore

if __name__ == '__main__':
    agent_loop()
