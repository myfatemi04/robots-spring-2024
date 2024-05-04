import datetime
import json
import os
from typing import List

import numpy as np
import PIL.Image
from openai import OpenAI
from vlms import image_message
from lmp_executor import StatefulLanguageModelProgramExecutor

'''
We construct a chat history using the system prompt and prev. observations
along with a notion of the task at hand
'''

with open("prompts/code_generation.txt") as f:
    code_generation_prompt = f.read()

class LanguageModelPlanner:
    def __init__(self, robot, instructions: str, root_log_dir: str = 'plan_logs'):
        self.instructions = instructions
        self.client = OpenAI()
        self.model = 'gpt-4-vision-preview'
        self.robot = robot
        self.history_simplified = [
            {'role': 'system', 'content': code_generation_prompt.replace("{INSTRUCTIONS}", instructions)},
        ]
        self.history = [
            {'role': 'system', 'content': code_generation_prompt.replace("{INSTRUCTIONS}", instructions)},
        ]
        self.code_executor = StatefulLanguageModelProgramExecutor()

        # choose an output directory
        now = datetime.datetime.now()
        log_name = f'{now.year}-{now.month:02d}-{now.day:02d}T{now.hour:02d}_{now.minute:02d}_{now.second:02d}'
        self.root_log_dir = root_log_dir
        self.log_name = log_name
        self.log_dir = os.path.join(root_log_dir, log_name)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "images"), exist_ok=True)

        print("Saving logs to", self.root_log_dir, "/", self.log_name)

    def save(self):
        serialized_history = []
        image_counter = 0

        def cvt_img_content(image_content):
            nonlocal image_counter

            image_counter += 1

            assert image_content['type'] == 'image'

            image_content['image'].save(os.path.join(self.log_dir, f'images/{image_counter}.png'))

            return {'type': 'image', 'image_url': f'./images/{image_counter}.png'}
        
        for msg in self.history:
            new_msg = {'role': msg['role']}
            if type(msg['content']) == list:
                new_msg_content = []
                for content in msg['content']:
                    if content['type'] == 'image':
                        new_msg_content.append(cvt_img_content(content))
                    else:
                        new_msg_content.append(content)
                new_msg['content'] = new_msg_content
            # sort of standardize the format
            elif type(msg['content']) == str:
                new_msg['content'] = [{'type': 'text', 'text': msg['content']}]
            elif msg['content']['type'] == 'image':
                new_msg['content'] = [cvt_img_content(msg['content'])]
            else:
                assert msg['content']['type'] == 'text'
                new_msg['content'] = [msg['content']]

            serialized_history.append(new_msg)

        # save messages
        with open(os.path.join(self.log_dir, "messages.json"), "w") as f:
            json.dump(serialized_history, f)


    def run_step(self, imgs: List[PIL.Image.Image], pcds: List[np.ndarray]):
        if True:
            response = self.client.chat.completions.create(
                model=self.model, messages=[
                    *self.history_simplified,
                    {'role': 'user', 'content': [
                        {"type": "text", "text": "This is the current observation. Please write your plan and code to complete the task. If you believe the task has been completed, call the robot.signal_completed() function."},
                        image_message(imgs[0])
                    ]}
                ]
            )

            content = response.choices[0].message.content
            print(content)

            self.history_simplified.append({
                # Avoid using too many credits on image inputs
                "role": "user", "content": "(Previous image observation)",
            })
            # We store this slightly different compared to OpenAI format
            self.history.append({
                'role': 'user', 'content': [
                    {"type": "text", "text": "This is the current observation."},
                    {"type": "image", "image": imgs[0]}
                ]
            })
            self.history_simplified.append({
                "role": "assistant", "content": content,
            })
            self.history.append({
                "role": "assistant", "content": content,
            })

            self.save()

            # find the code block in the language model's response
            code_start = content.find("```python") + 9
            if code_start != -1:
                code_end = content.find("```", code_start)
            else:
                code_end = -1
            if code_start == -1 or code_end == -1:
                print("ERROR: Could not find code block in response")
                # just add a warning message in the history and hopefully the llm will correct its mistake next time
                self.history.append({"role": "system", "content": "Please include a properly-formatted code block that begins with \"```python\" and ends with \"```\"."})
                self.history_simplified.append({"role": "system", "content": "Please include a properly-formatted code block that begins with \"```python\" and ends with \"```\"."})
                input("Operator press enter to continue execution.")
                return
            
            code = content[code_start:code_end]
        
        print("Extracted code segment:")
        print(code)
        if 'y' != input("OK to execute this code? (y/n): "):
            print("Ignoring this code.")
            return

        # from here, we create the Scene object, which we can pass to the LM planner
        from lmp_scene_api import Scene

        scene = Scene(imgs, pcds, [f'img{i}' for i in range(1, 1 + len(imgs))], self, 'vlm')

        # run the code
        self.code_executor.update(scene=scene, robot=self.robot)
        success, stack_trace = self.code_executor.execute(code)
        print(success)
        print(stack_trace)

# test the LMP planner
def main():
    from rgbd import RGBD
    from lmp_scene_api import Robot
    from rotation_utils import vector2quat
    import matplotlib

    matplotlib.use("Qt5agg") # required so we don't crash for some reason

    rgbd = RGBD(num_cameras=1)
    robot = Robot(polymetis_server_ip='<mock>')
    planner = LanguageModelPlanner(robot, "Put a piece of candy in the cup")

    print("Resetting robot position...")
    robot.move_to([0.4, 0, 0.4], orientation=vector2quat(claw=[0, 0, -1], right=[0, -1, 0]))
    robot.start_grasp()
    robot.stop_grasp()
    print("Robot position reset.")

    try:
        rgbs, pcds = rgbd.capture()
        imgs = [PIL.Image.fromarray(rgb) for rgb in rgbs]

        planner.run_step(imgs, pcds)
    finally:
        rgbd.close()

if __name__ == '__main__':
    main()