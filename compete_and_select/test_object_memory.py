import PIL.Image
from lmp_planner import LanguageModelPlanner
from lmp_scene_api import Scene, Robot
from rgbd import RGBD

robot = Robot('<mock>')
rgbd = RGBD(num_cameras=1)

try:
    vlm_planner = LanguageModelPlanner(robot, instructions='Choose a piece of candy for the user.')

    rgbs, pcds = rgbd.capture()
    imgs = [PIL.Image.fromarray(rgb) for rgb in rgbs]

    scene = Scene(imgs, pcds, [f'img{i}' for i in range(1, 1 + len(imgs))], vlm_planner, 'vlm')
    scene.choose('a piece of candy', 'for the user to eat')

    ### Memory should be updated between these steps ###

    scene.choose('a piece of candy', 'for the user to eat')

finally:
    rgbd.close()
