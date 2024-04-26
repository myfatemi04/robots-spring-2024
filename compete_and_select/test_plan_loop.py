# we run a loop to control the robot with a VLM

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import segment_point_cloud
from generate_object_candidates import detect, draw_set_of_marks
from detect_grasps import detect_grasps
from panda import Panda
from rgbd import RGBD
from scipy.spatial.transform import Rotation
from transformers import SamModel, SamProcessor

model = SamModel.from_pretrained("facebook/sam-vit-base").cuda()
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def vector2quat(claw, right=None):
    claw = claw / np.linalg.norm(claw)
    right = right / np.linalg.norm(right)

    palm = np.cross(right, claw)
    matrix = np.array([
        [palm[0], right[0], claw[0]],
        [palm[1], right[1], claw[1]],
        [palm[2], right[2], claw[2]],
    ])
    
    return Rotation.from_matrix(matrix).as_quat()

def matrix2quat(matrix):
    return Rotation.from_matrix(matrix).as_quat()

def main():
    panda = Panda()
    # select camera IDs to use
    rgbd = RGBD(num_cameras=1)

    matplotlib.use("Qt5agg")
    plt.ion()

    # used to store persistent data
    vars = {}

    locals_ = locals()

    pcd_segmenter = segment_point_cloud.SamPointCloudSegmenter(device='cuda', render_2d_results=False)

    try:
        print("Resetting robot position...")
        panda.move_to([0.4, 0, 0.4], orientation=vector2quat(claw=[0, 0, -1], right=[0, -1, 0]))
        panda.start_grasp()
        panda.stop_grasp()
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
            
            # Render the detections.
            plt.clf()
            draw_set_of_marks(imgs[0], detects, live=True)
            fig = plt.gcf()
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.001)

            object_number = int(input("Number of object to grasp:"))
            if object_number < 0:
                break

            box = detects[object_number - 1]['box']
            box = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]

            start_time = time.time()
            segmented_pcd_xyz, segmented_pcd_color, _segmentation_masks = pcd_segmenter.segment(imgs[0], pcds[0], box, imgs[1:], pcds[1:])
            end_time = time.time()

            # render the segmented point cloud
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_title(f"3D Segmentation")
            ax.scatter(segmented_pcd_xyz[:, 0], segmented_pcd_xyz[:, 1], segmented_pcd_xyz[:, 2], c=segmented_pcd_color/255.0, s=0.5)

            grasps = detect_grasps(
                segmented_pcd_xyz,
                segmented_pcd_color,
                voxel_size=0.005,
                min_points_in_voxel=2,
                gripper_width=0.2,
                max_alpha=15,
                hop_size=1,
                window_size=2,
                top_k_per_angle=5,
                show_rotated_voxel_grids=False
            )

            flattest_grasp = None
            best_score = None

            for grasp in grasps:
                _worst_alpha, start, end = grasp
                (x1, y1, z1) = (start)# - lower_bound) / voxel_size
                (x2, y2, z2) = (end)# - lower_bound) / voxel_size

                score = -_worst_alpha # -abs(z2 - z1) / (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
                if flattest_grasp is None or score > best_score:
                    flattest_grasp = grasp
                    best_score = score

                ax.scatter(x1, y1, z1, c='r')
                ax.scatter(x2, y2, z2, c='g')
                ax.plot([x1, x2], [y1, y2], [z2, z2], c='b')

            fig = plt.gcf()
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.001)

            input("[press enter to continue]")

            if flattest_grasp is not None:
                print("A grasp has been identified.")
                _, start, end = flattest_grasp
                centroid = np.array([
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2,
                    (start[2] + end[2]) / 2,
                ])
                # uses the vector conventions from vector2quat
                right = [end[0] - start[0], end[1] - start[1], 0]
                claw = [0, 0, -1]
                print("Claw:", claw)
                print("Right:", right)
                print("Start:", start)
                print("End:", end)
                target_rotation_quat = vector2quat(claw, right)
                print(f"Centroid: {centroid[0]:.2f} {centroid[1]:.2f} {centroid[2]:2f}")
                print(f"Target rotation:")
                print(Rotation.from_quat(target_rotation_quat).as_matrix())

                if 'y' == input("Go to this position? (y/n) "):
                    panda.move_to(centroid, orientation=target_rotation_quat)

                # now we grasp
                panda.start_grasp()
                # and move to base position
                panda.move_to([0.4, 0, 0.4], orientation=vector2quat(claw=[0, 0, -1], right=[0, -1, 0]))

    except KeyboardInterrupt:
        pass
    finally:
        plt.close()
        rgbd.close()

if __name__ == '__main__':
    main()
