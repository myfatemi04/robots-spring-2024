# Creates a command line for controlling the robot.
# Ideally, if we can control the robot entirely using a command line, an LLM will be able to too.

import time
import threading
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import traceback

from panda import Panda
from rgbd import RGBD

def euler2quat(yaw, pitch, roll):
    rot = Rotation.from_euler('xyz', [yaw, pitch, roll], degrees=True)
    # scalar-last
    rot_quat = rot.as_quat()
    # make scalar-first
    # return np.array([rot_quat[-1], *rot_quat[:-1]])
    return rot_quat

def main():
    panda = Panda()
    # select camera IDs to use
    rgbd = RGBD()

    matplotlib.use("Qt5agg")
    plt.ion()

    # used to store persistent data
    vars = {}

    locals_ = locals()

    def handle_commands():
        nonlocal locals_

        while True:
            try:
                command = input()
                if command.startswith("!"):
                    command = command[1:]
                    print(eval(command, globals(), locals_))
                else:
                    exec(command, globals(), locals_)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(">>> Exception:")
                print(e)
                traceback.print_exc()

    handle_commands_thread = threading.Thread(target=handle_commands, daemon=True)
    handle_commands_thread.start()

    try:
        while True:
            start_time = time.time()
            rgbs, pcds = rgbd.capture()
            end_time = time.time()
            # print(f"Capture duration: {end_time - start_time:.3f}s")

            locals_ = locals()

            plt.clf()
            plt.subplot(1, 2, 1)
            plt.title("Camera 0")
            plt.imshow(rgbs[0])
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title("Camera 1")
            plt.imshow(rgbs[1])
            plt.axis('off')
            # Prevents matplotlib from stealing focus.
            fig = plt.gcf()
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.001)
    except KeyboardInterrupt:
        plt.close()
        rgbd.close()

if __name__ == '__main__':
    main()
