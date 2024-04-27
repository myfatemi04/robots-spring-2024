import datetime
import json
import os
from typing import List

import numpy as np
import PIL.Image
from openai import OpenAI
from vlms import image_message

'''
We construct a chat history using the system prompt and prev. observations
along with a notion of the task at hand
'''

with open("prompts/code_generation.txt") as f:
    code_generation_prompt = f.read()

class LanguageModelPlanner:
    def __init__(self, instructions, root_log_dir='plan_logs'):
        self.client = OpenAI()
        self.model = 'gpt-4-vision-preview'
        self.history_simplified = [
            {'role': 'system', 'content': code_generation_prompt.replace("{INSTRUCTIONS}", instructions)},
        ]
        self.history = [
            {'role': 'system', 'content': code_generation_prompt.replace("{INSTRUCTIONS}", instructions)},
        ]

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
        # from lmp_scene_api import Scene

        response = self.client.chat.completions.create(
            model=self.model, messages=[
                *self.history_simplified,
                {'role': 'user', 'content': [
                    {"type": "text", "text": "This is the current observation."},
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

        self.save()

        # from here, we create the Scene object, which we can pass to the LM planner

# test the LMP planner
def main():
    from rgbd import RGBD
    
    rgbd = RGBD(num_cameras=1)
    planner = LanguageModelPlanner("Put the blue block into the white cup")

    try:
        rgbs, pcds = rgbd.capture()
        imgs = [PIL.Image.fromarray(rgb) for rgb in rgbs]

        planner.run_step(imgs, pcds)
    finally:
        rgbd.close()

if __name__ == '__main__':
    main()
