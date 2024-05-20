# we run a loop to control the robot with a VLM

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from lmp_executor import StatefulLanguageModelProgramExecutor
from lmp_scene_api import Robot, Scene
from rgbd import RGBD
from rotation_utils import vector2quat
from transformers import SamModel, SamProcessor
from sam import sam_model as model, sam_processor as processor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    robot = Robot()
    # select camera IDs to use
    rgbd = RGBD(num_cameras=1)

    matplotlib.use("Qt5agg")
    plt.ion()
    
    code_executor = StatefulLanguageModelProgramExecutor()

    try:
        print("Resetting robot position...")
        robot.move_to([0.4, 0, 0.4], orientation=vector2quat(claw=[0, 0, -1], right=[0, -1, 0]))
        robot.start_grasp()
        robot.stop_grasp()
        print("Robot position reset.")

        while True:
            start_time = time.time()
            rgbs, pcds = rgbd.capture()
            end_time = time.time()

            imgs = [PIL.Image.fromarray(rgb) for rgb in rgbs]

            plt.clf()
            # plt.subplot(1, 2, 1)
            plt.title("Camera 0")
            plt.imshow(rgbs[0])
            plt.axis('off')
            # plt.subplot(1, 2, 2)
            # plt.title("Camera 1")
            # plt.imshow(rgbs[1])
            plt.axis('off')
            # Prevents matplotlib from stealing focus.
            fig = plt.gcf()
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.001)

            if any([pcd is None for pcd in pcds]):
                continue

            # For now, will have no control flow (if statements, etc.)
            # Any decisions will need to be made by the VLM for what robot
            # action to take next. If we want to loop until all cubes are
            # picked up for example, we will need to manually check that
            # the table is clear first. This will use a lot of tokens
            # but it will be a nice sanity check.

            # I have created a nice `Scene` class which will hopefully make it easy to manipulate the environment
            scene = Scene(imgs, pcds, [f'cam{i+1}' for i in range(len(imgs))])
            block = scene.choose('block')
            cup = scene.choose('cup')
            print("Cup centroid:")
            print(cup.centroid)
            robot.grasp(block)
            hover_pos = np.array(cup.centroid)
            hover_pos[2] += 0.2
            robot.move_to(hover_pos)
            robot.release()

    except KeyboardInterrupt:
        pass
    finally:
        plt.close()
        rgbd.close()

if __name__ == '__main__':
    main()
