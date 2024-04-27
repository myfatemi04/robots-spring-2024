from lmp_scene_api import Scene
from vlms import gpt4v_plusplus

'''
We construct a chat history using the system prompt and prev. observations
along with a notion of the task at hand
'''

class LanguageModelPlanner:
    def __init__(self):
        self.history = []

    def run_step(self, obs: Scene):
        gpt4v_plusplus([
            ("Look at this image.", obs.imgs[0])
        ])

