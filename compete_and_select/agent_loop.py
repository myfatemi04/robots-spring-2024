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

import dotenv

dotenv.load_dotenv()

import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from openai import OpenAI

from .agent_state import AgentState
from .config import Config
from .event_stream import (CodeActionEvent, EventStream, ExceptionEvent,
                           ReflectionEvent, VerbalFeedbackEvent,
                           VisualPerceptionEvent)
from .lmp_planner import (StatefulLanguageModelProgramExecutor,
                          reason_and_generate_code)
from .lmp_scene_api import Human, Robot, Scene
from .memory_bank_v2 import MemoryBank
from .perception.rgbd import RGBD
from .perception.rgbd_asynchronous_tracker import RGBDAsynchronousTracker
from .rotation_utils import vector2quat
from .vlms import image_message

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
            if event.prompt:
                context.append({'role': 'assistant', 'content': f'{event.prompt}'})
            context.append({'role': 'user', 'content': f'{event.text}'})
        elif isinstance(event, CodeActionEvent):
            context.append({
                'role': 'assistant',
                'content': event.raw_content,
            })
        elif isinstance(event, ExceptionEvent):
            context.append({
                'role': 'system',
                'content': f"Exception ({event.exception_type}): {event.text}"
            })
    return context

def create_vision_model_context(event_stream: EventStream, max_vision_events_to_include=1):
    """
    We predict P(useful description | visual input and prior dialogue)
    """
    vision_events = [i for i, event in enumerate(event_stream.events) if isinstance(event, VisualPerceptionEvent)]
    vision_event_cutoff = vision_events[-max_vision_events_to_include]
    context = []
    for i, event in enumerate(event_stream.events):
        if isinstance(event, VisualPerceptionEvent):
            if i >= vision_event_cutoff:
                context.append({
                    'role': 'system',
                    'content': [
                        {'type': 'text', 'text': "Here is what you currently see."},
                        image_message(event.imgs[0]), # type: ignore
                    ]
                })
            else:
                context.append({
                    'role': 'system',
                    'content': '<Prior observation>'
                })
        elif isinstance(event, ReflectionEvent):
            context.append({
                'role': 'assistant',
                'content': f"Reflection: {event.reflection}"
            })
        elif isinstance(event, VerbalFeedbackEvent):
            if event.prompt:
                context.append({'role': 'assistant', 'content': f'{event.prompt}'})
            context.append({'role': 'user', 'content': f'{event.text}'})
        elif isinstance(event, CodeActionEvent):
            context.append({
                'role': 'assistant',
                'content': event.raw_content,
            })
        elif isinstance(event, ExceptionEvent):
            context.append({
                'role': 'system',
                'content': f"Exception ({event.exception_type}): {event.text}"
            })
    return context

def agent_loop():
    event_stream = EventStream()
    memory_bank = MemoryBank()
    
    config = Config(
        use_xmem=False,
        use_visual_cot=True,
    )
    
    ### Initialize camera capture. ###
    # Calibration needs to be done in the main thread, so if we use the asynchronous
    # object tracker, we must disable calibration from being called automatically
    # during each capture.
    rgbd = RGBD(num_cameras=1, auto_calibrate=False)
    # rgbd = RGBD(num_cameras=2, camera_ids=['000259521012', '000243521012'], auto_calibrate=False)
    # allows frames to be tracked even when work is being done on the main thread.
    # this should increase the quality of object tracking.
    tracker = RGBDAsynchronousTracker(rgbd, disable_tracking=not config.use_xmem)
    tracker.open()

    ### Initialize agent_state. ###
    agent_state = AgentState(event_stream, memory_bank, tracker, config)
    human = Human(agent_state)
    code_executor = StatefulLanguageModelProgramExecutor(vars={"np": np, "human": human})

    client = OpenAI()

    robot = Robot('192.168.1.222')

    if '--no-reset' not in sys.argv:
        robot.start_grasp()
        robot.stop_grasp()
        robot.move_to([0.4, 0, 0.4], orientation=vector2quat(claw=[0, 0, -1], right=[0, -1, 0]))

    input("> Ready. >")

    # Wait for calibration

    if not os.path.exists("calibrations.pkl"):
        has_pcd = False
        while not has_pcd:
            # uses a threading.Event to wait for next frame
            (rgbs, pcds, _) = tracker.next()
            for i in range(len(rgbs)):
                tracker.rgbd.try_calibrate(i, rgbs[i])
            # rgbs, pcds = rgbd.capture()
            # print("Calibrated:", calibrated)
            has_pcd = all(pcd is not None for pcd in pcds)
            plt.title("Camera 1")
            plt.imshow(rgbs[1])
            plt.pause(0.05)

        # save calibration
        calibrations = [rgbd.cameras[0].extrinsic_matrix, rgbd.cameras[1].extrinsic_matrix]
        with open("calibrations.pkl", "wb") as f:
            pickle.dump(calibrations, f)
        
        print("Saved calibrations.")
    else:
        with open("calibrations.pkl", "rb") as f:
            calibrations = pickle.load(f)
            for i, calibration in enumerate(calibrations[:len(rgbd.cameras)]):
                rgbd.cameras[i].extrinsic_matrix = calibration

        print("Restored calibrations.")

    event_stream.write(VerbalFeedbackEvent(input("Instructions for robot: ")))

    rgbs, pcds = rgbd.capture()
    imgs = [PIL.Image.fromarray(rgb) for rgb in rgbs]
    scene = Scene(imgs, pcds, agent_state)
    event_stream.write(VisualPerceptionEvent(scene.imgs, scene.pcds))
    
    try:
        for i in range(5):
            context = create_primary_context(event_stream)
            rationale, code, raw_content = reason_and_generate_code(context, client)
            event_stream.write(CodeActionEvent(rationale, code, raw_content))

            print("Reasoning and code generation:")
            print(raw_content)

            if code is not None:
                code_executor.update(scene=scene, robot=robot)
                status, output = code_executor.execute(code)
                if not status:
                    print(f"Error: {output}")
                    event_stream.write(ExceptionEvent("CodeExecutionError", output)) # type: ignore

            time.sleep(1)

            # Add new item to visual memory.
            # Ask the robot to compare / contrast the images.
            rgbs, pcds = rgbd.capture()
            imgs = [PIL.Image.fromarray(rgb) for rgb in rgbs]
            scene = Scene(imgs, pcds, agent_state)
            event_stream.write(VisualPerceptionEvent(scene.imgs, scene.pcds))

            # Display the new data we received.
            # Also display tracked objects, and their names.
            # We will need a notion of an object registry.
            # Perhaps if an object has not been seen for [X]
            # amount of frames, we deleted it from the registry.
            # What other information is needed to introduce object
            # permanence? What ways are there to introduce object
            # memories? Maybe we simply enumerate the objects in
            # the scene and record changes that occur with them.
            # Surely this capability is useful!
            plt.title("Camera 0")
            plt.imshow(rgbs[0])
            # Plot masks.

            plt.pause(0.05)

            # Ask agent to reflect on the past two timesteps.
            ctx = create_vision_model_context(event_stream, max_vision_events_to_include=2)
            ctx.append({
                "role": "system",
                "content":
                    "Reflect on your most recent action. What happened between the last two observations? "
                    "Write a list of the changes that occurred, and then summarize the changes. "
                    "Alternatively, if you encounter an error, consider asking the human to select an object."
            })
            # Now, we reflect on the difference between the previous and current action.
            cmpl = client.chat.completions.create(model='gpt-4-vision-preview', messages=ctx)
            reflection = cmpl.choices[0].message.content
            print("Reflection:")
            print(reflection)
            event_stream.write(ReflectionEvent(reflection))

    except Exception as e:
        print("Error:", e)
    finally:
        robot.stop_grasp()
        tracker.close()

        # Save the event stream
        i = 0
        while os.path.exists(f"./memories/event_stream_{i}.pkl"):
            i += 1
        
        print("Run ID:", i)

        with open(f"memories/event_stream_{i}.pkl", "wb") as f:
            pickle.dump(event_stream, f)

if __name__ == '__main__':
    agent_loop()
