# Creates a command line for controlling the robot.
# Ideally, if we can control the robot entirely using a command line, an LLM will be able to too.

import pickle
import threading
import time
import traceback

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from panda import Panda
from compete_and_select.perception.rgbd import RGBD
from scipy.spatial.transform import Rotation
from transformers import SamModel, SamProcessor
from sam import sam_model as model, sam_processor as processor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vector2quat(claw, right):
    claw = claw / np.linalg.norm(claw)
    right = right / np.linalg.norm(right)

    palm = np.cross(right, claw)
    matrix = np.array([
        [palm[0], right[0], claw[0]],
        [palm[1], right[1], claw[1]],
        [palm[2], right[2], claw[2]],
    ])
    
    return Rotation.from_matrix(matrix).as_quat() # type: ignore

def matrix2quat(matrix):
    return Rotation.from_matrix(matrix).as_quat() # type: ignore

def get_sam_mask(image, input_boxes):
    with torch.no_grad():
        inputs = processor(image, input_boxes=input_boxes, return_tensors="pt").to(device)
        outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()) # type: ignore
        scores = outputs.iou_scores

    plt.title("SAM detections")
    plt.imshow(image)
    for i in range(len(masks[0][0])):
        plt.imshow(masks[0][0][i], alpha=masks[0][0][i] * scores[0, 0, i].cpu())

    print("Number of masks:", len(masks[0][0]))
    print("Mask scores:", scores[0, 0])

    # Prevents matplotlib from stealing focus.
    plt.ioff()
    fig = plt.gcf()
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.001)
    plt.show()
    plt.ion()

def main():
    panda = Panda()
    # select camera IDs to use
    rgbd = RGBD()

    matplotlib.use("Qt5agg")
    plt.ion()

    # used to store persistent data
    vars = {}

    locals_ = locals()

    def save(object, filename):
        with open(filename, "wb") as f:
            pickle.dump(object, f)

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


            locals_ = locals()

            if 'sam_img' in vars and 'sam_pts' in vars:
                plt.close()
                print("Generating SAM mask.")
                get_sam_mask(vars['sam_img'], vars['sam_pts'])
                del vars['sam_img']
                del vars['sam_pts']

            # if pcds[0] is not None:
            #     print(get_normal_map(pcds[0])[0][471,897])
            #     break

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
        pass
    finally:
        plt.close()
        rgbd.close()

if __name__ == '__main__':
    main()
