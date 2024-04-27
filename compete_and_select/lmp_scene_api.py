# from profile import profile
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import segment_point_cloud
from detect_grasps import detect_grasps
from generate_object_candidates import detect as detect_objects_2d
from generate_object_candidates import draw_set_of_marks
from panda import Panda
from rotation_utils import vector2quat
from scipy.spatial.transform import Rotation
from functools import cached_property
from lmp_planner import LanguageModelPlanner
from openai import OpenAI
from vlms import image_message

pcd_segmenter = segment_point_cloud.SamPointCloudSegmenter(device='cuda', render_2d_results=False)

class Object:
    def __init__(self, point_cloud, colors, clip_features=None):
        # may just store objects in world frame to avoid having to keep track
        # of rotations and such
        self.point_cloud = point_cloud
        self.colors = colors
        self.clip_features = clip_features

    @cached_property
    def centroid(self):
        return self.point_cloud.mean(axis=0)

    # at some point expose this to the model so it can generate grasps
    # and then rank them later
    def generate_grasps(self):
        # will need to add some kind of collision detection here next
        grasps = detect_grasps(
            self.point_cloud,
            self.colors,
            voxel_size=0.005,
            min_points_in_voxel=2,
            gripper_width=0.2,
            max_alpha=15,
            hop_size=1,
            window_size=2,
            top_k_per_angle=5,
            show_rotated_voxel_grids=False
        )
        return grasps

class Scene:
    def __init__(self, imgs, pcds, view_names, lmp_planner=None, selection_method_for_choose=None):
        self.imgs: List[PIL.Image.Image] = imgs
        self.pcds: List[np.ndarray] = pcds
        self.view_names = view_names
        self.selection_method_for_choose = selection_method_for_choose or ('human' if lmp_planner is None else 'vlm')
        self.lmp_planner = lmp_planner

    def _choose_with_vlm(self, annotated_image, object_type, purpose):
        client = OpenAI()
        task = 'put the blue block into the white cup'
        completion = client.chat.completions.create([
            {"role": "system", "content":
             "You are an assistant which helps robots make motion plans. In each image, a set of objects the robot could feasibly interact with "
             "are shown with bounding boxes. Next to each bounding box is a number which uniquely identifies the object. You help the robot choose "
             "particular objects to interact with by outputting the number of the object."
            },
            {"role": "user", "content": [
                image_message(annotated_image),
                {"type": "text", f"content": f"Task: {task}\nYou are now selecting {object_type} (particularly, {purpose}). What object ID should be selected here? Write 'Object ID: (number 1...n)'."}
            ]}
        ])
        content: str = completion.choices[0].content
        print("Set-of-marks prompt response:")
        print(content)
        index = content.lower().find("id: ")
        object_id_str = content[index + 4:].strip()
        object_id_str = object_id_str[:5]
        # quick and dirty trick
        object_id_str = ''.join([c for c in object_id_str if c in '0123456789'])
        print("Object ID string:", object_id_str)
        object_id = int(object_id)
        return object_id
    
    def choose(self, object_type, purpose):
        """
        <lm-docs>
        Allows the robot to choose an object. This returns an `Object` variable,
        which can be `.grasp()`ed with the `robot` class.
        </lm-docs>
        
        Implements set-of-marks prompting (labeling bounding boxes with numbers).
        Precisely, this method:
        1. Locates the objects of type `object_name` that are in the image
        2. Labels these objects with numbers and prompts the selector (human or LM)
        to respond with the number of the object they wish to select.
        """
        base_image_index = 0

        base_rgb_image = self.imgs[base_image_index]
        base_point_cloud = self.pcds[base_image_index]
        nviews = len(self.imgs)
        supplementary_rgb_images = [self.imgs[i] for i in range(nviews) if i != base_image_index]
        supplementary_point_clouds = [self.pcds[i] for i in range(nviews) if i != base_image_index]

        image = self.imgs[base_image_index]
        detections_2d = detect_objects_2d(image, object_type)

        if len(detections_2d) == 0:
            print("No object candidates detected")
            return None

        if self.selection_method_for_choose == 'human':
            plt.title(f"Please choose: {purpose}")
            draw_set_of_marks(image, detections_2d, live=True)
            plt.show()

            object_id = int(input(f"Choice (choose a number 1 to {len(detections_2d)}, or -1 to cancel): "))

        elif self.selection_method_for_choose == 'vlm':
            annotated_image = draw_set_of_marks(image, detections_2d)

            object_id = self._choose_with_vlm(annotated_image, object_type, purpose)
            
            raise NotImplemented
        
        if object_id == -1:
            print("Cancelled by selector")
            return None
        
        if not (1 <= object_id <= len(detections_2d)):
            print("Object choice out of range")
            return None
        
        detection = detections_2d[object_id - 1]
        box = detection['box']
        x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        segmented_pcd_xyz, segmented_pcd_color, _segmentation_masks = \
            pcd_segmenter.segment(base_rgb_image, base_point_cloud, [x1, y1, x2, y2], supplementary_rgb_images, supplementary_point_clouds)
        
        return Object(segmented_pcd_xyz, segmented_pcd_color, clip_features=None)

    def locate(self, object_name, target=0):
        detections_2d = detect_objects_2d(self.imgs[target], object_name)

        base_rgb_image = self.imgs[target]
        base_point_cloud = self.pcds[target]
        supplementary_rgb_images = [self.imgs[i] for i in range(2) if i != target]
        supplementary_point_clouds = [self.pcds[i] for i in range(2) if i != target]

        objects: List[Object] = []

        for i, detection in enumerate(detections_2d):
            box = detection['box']
            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            segmented_pcd_xyz, segmented_pcd_color, _segmentation_masks = \
                pcd_segmenter.segment(base_rgb_image, base_point_cloud, [x1, y1, x2, y2], supplementary_rgb_images, supplementary_point_clouds)
            
            objects.append(Object(segmented_pcd_xyz, segmented_pcd_color, clip_features=None))

        return objects

class Robot(Panda):
    grasping = False

    def grasp(self, object: Object):
        assert not self.grasping, "Robot is currently in a `grasping` state, and cannot grasp another object. If you believe this is in error, call `robot.release()`."

        # use simplest grasping metric: lowest alpha [surface misalignment] score
        grasps = object.generate_grasps() # each is (alpha, start, end)

        best_grasp = None
        best_score = None

        for grasp in grasps:
            _worst_alpha, start, end = grasp
            (x1, y1, z1) = (start)# - lower_bound) / voxel_size
            (x2, y2, z2) = (end)# - lower_bound) / voxel_size

            score = -_worst_alpha # -abs(z2 - z1) / (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if best_grasp is None or score > best_score:
                best_grasp = grasp
                best_score = score

        start = np.array(start)
        end = np.array(start)
        centroid = (start + end)/2
        right = (end - start)
        right /= np.linalg.norm(right)
        
        print("A grasp has been identified.")
        _, start, end = best_grasp
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
        input("operator press enter to confirm.")

        self.move_to(centroid, orientation=target_rotation_quat)
        self.start_grasp()
        self.grasping = True

    def release(self):
        if not self.grasping:
            print("Calling `robot.release()` when not in a grasping state")
        self.grasping = False
        self.stop_grasp()
