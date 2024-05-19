from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import segment_point_cloud
from agent_state import AgentState
from clients import vlm_client
from detect_objects import Detection
from detect_objects import detect as detect_objects_2d
from event_stream import (CodeActionEvent, ObjectSelectionDetectionResult,
                          ObjectSelectionInitiation,
                          ObjectSelectionPolicyCreation,
                          ObjectSelectionPolicySelection, ReflectionEvent,
                          VerbalFeedbackEvent, VisualPerceptionEvent)
from lmp_planner import LanguageModelPlanner
from lmp_scene_api_object import Object
from memory_bank_v2 import Memory, MemoryKey, Retrieval
from object_detection_utils import draw_set_of_marks
from openai import OpenAI
from panda import Panda
from rotation_utils import vector2quat
from scipy.spatial.transform import Rotation
from select_object_v2 import format_object_detections
from vlms import image_message

pcd_segmenter = segment_point_cloud.SamPointCloudSegmenter(render_2d_results=False)



def create_get_selection_policy_context(agent_state: AgentState, detections: List[Detection], retrievals: List[List[Retrieval]], annotated_image):
    last_vision_event = [i for i, event in enumerate(agent_state.event_stream.events) if isinstance(event, VisualPerceptionEvent)][-1]
    context = []

    context.append({
        "role": "system",
        "content": "You are an assistive tool which helps robots make motion plans."
    })

    last_object_selection_initiation = [i for i, event in enumerate(agent_state.event_stream.events) if isinstance(event, ObjectSelectionInitiation)][-1]

    for i, event in enumerate(agent_state.event_stream.events):
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
                        image_message(annotated_image), # type: ignore
                    ]
                })
        elif isinstance(event, ReflectionEvent):
            context.append({
                'role': 'assistant',
                'content': event.reflection,
            })
        elif isinstance(event, VerbalFeedbackEvent):
            context.append({
                'role': 'user',
                'content': event.text,
            })
        elif isinstance(event, CodeActionEvent):
            context.append({
                'role': 'assistant',
                'content': event.raw_content,
            })
        elif isinstance(event, ObjectSelectionInitiation) and i == last_object_selection_initiation:
            object_detections_string = format_object_detections(detections, retrievals)

            content = f"""It is now time for you to select an object to interact with. Object type: {event.object_type}. Object purpose: {event.object_purpose}.
            
The following objects have been detected:
{object_detections_string}

Rank the objects in the scene according to how likely they are to be the best choice.
Respond with 'likely', 'neutral', and 'unlikely' for each object. Format your response as follows:
```
Reasoning:
(Your reasoning goes here)

Choices:
1: likely
2: neutral
3: unlikely

Answer as if you are controlling a hypothetical robot.
"""
            context.append({
                'role': 'system',
                'content': content,
            })
    return context

def parse_likelihood_response(response: str) -> Tuple[str, Dict[int, str]]:
    """
    FORMAT:
    ```
    Reasoning:
    The object is commonly used in this context.

    Choices:
    1: likely
    2: neutral
    3: unlikely
    ```
    """

    # Parse the response
    choices_index = response.find('Choices:')
    reasoning = response[:choices_index].strip()
    choices_str = response[choices_index + 8:].strip()

    # Extract individual choices
    choices = {}
    for line in choices_str.strip().split('\n'):
        if ':' in line:
            obj_num, choice = line.split(':')
            choices[int(obj_num.strip())] = choice.strip()
        else:
            # break at the first line without an object.
            break

    return (reasoning, choices)

def get_selection_policy(context: list):
    """
    Scope of this function:
     - Given an input state, output a policy for which objects to select
    """

    response = vlm_client.chat.completions.create(
        model='gpt-4-vision-preview',
        messages=[*context]
    ).choices[0].message.content
    assert response is not None

    print("Created object selection policy:")
    print(response)

    reasoning, choices = parse_likelihood_response(response)

    # get logits
    logits = np.zeros(max(choices.keys()))
    for key, value in choices.items():
        if value.lower().strip() == 'unlikely':
            logits[key - 1] = -1
        elif value.lower().strip() == 'likely':
            logits[key - 1] = 1

    return (reasoning, logits)

class Scene:
    def __init__(self, imgs, pcds, agent_state: AgentState):
        self.imgs: List[PIL.Image.Image] = imgs
        self.pcds: List[Optional[np.ndarray]] = pcds or [None] * len(imgs)
        self.agent_state = agent_state

    def choose(self, object_type, purpose):
        self.agent_state.event_stream.write(ObjectSelectionInitiation(object_type, purpose))

        detections = detect_objects_2d(self.imgs[0], object_type)
        print("Number of detections:", len(detections))

        self.agent_state.event_stream.write(ObjectSelectionDetectionResult(detections))

        retrievals = [self.agent_state.memory_bank.retrieve(detection.embedding, threshold=0.5) for detection in detections]

        annotated_image = draw_set_of_marks(self.imgs[0], detections)

        rationale, logits = get_selection_policy(create_get_selection_policy_context(self.agent_state, detections, retrievals, annotated_image))

        self.agent_state.event_stream.write(ObjectSelectionPolicyCreation(rationale, logits))

        print("Logits:", logits)
        plt.title("Annotated image")
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.show()

        selected_object_id = np.argmax(logits)

        detection = detections[selected_object_id]
        box = detection['box']
        x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        segmented_pcd_xyz, segmented_pcd_color, _segmentation_masks = \
            pcd_segmenter.segment(self.imgs[0], self.pcds[0], [x1, y1, x2, y2], self.imgs[1:], self.pcds[1:])
        
        selected_object = Object(segmented_pcd_xyz, segmented_pcd_color, clip_features=None)

        self.agent_state.event_stream.write(ObjectSelectionPolicySelection(selected_object_id, selected_object))

        return selected_object

class Scene_v0:
    def __init__(self, imgs, pcds=None, view_names=None, lmp_planner: Optional[LanguageModelPlanner] = None, selection_method_for_choose=None):
        self.imgs: List[PIL.Image.Image] = imgs
        self.pcds: List[Optional[np.ndarray]] = pcds or [None] * len(imgs)
        self.view_names = view_names or [f'img{i}' for i in range(len(imgs))]
        self.selection_method_for_choose = selection_method_for_choose or ('human' if lmp_planner is None else 'vlm')
        self.lmp_planner = lmp_planner

    def _choose_with_vlm(self, annotated_image, object_type, purpose):
        assert self.lmp_planner is not None
        client = OpenAI()
        task = self.lmp_planner.instructions
        completion = client.chat.completions.create(model='gpt-4-vision-preview', messages=[
            {"role": "system", "content":
             "You are an assistant which helps robots make motion plans. In each image, a set of objects the robot could feasibly interact with "
             "are shown with bounding boxes. Next to each bounding box is a number which uniquely identifies the object. You help the robot choose "
             "particular objects to interact with by outputting the number of the object."
            },
            {"role": "user", "content": [
                image_message(annotated_image), # type: ignore
                {"type": "text", "text": f"Task: {task}\nSelect the box corresponding to {purpose}. What object ID should be selected here? Write 'Object ID: (number 1...n)'."}
            ]}
        ], temperature=0)
        content = completion.choices[0].message.content
        assert content is not None
        print("Set-of-marks prompt response:")
        print(content)
        index = content.lower().find("id: ")
        object_id_str = content[index + 4:].strip()
        object_id_str = object_id_str[:5]
        # quick and dirty trick
        object_id_str = ''.join([c for c in object_id_str if c in '0123456789'])
        print("Object ID string:", object_id_str)
        object_id = int(object_id_str)
        return object_id

    def choose_v1(self, object_type, purpose):
        base_image_index = 0

        base_rgb_image = self.imgs[base_image_index]
        base_point_cloud = self.pcds[base_image_index]
        nviews = len(self.imgs)
        supplementary_rgb_images = [self.imgs[i] for i in range(nviews) if i != base_image_index]
        supplementary_point_clouds = [self.pcds[i] for i in range(nviews) if i != base_image_index]

        detections_2d = detect_objects_2d(base_rgb_image, object_type)

        if len(detections_2d) == 0:
            print("<NO DETECTIONS>")
            return None
        
        assert self.lmp_planner is not None
        
        # ObjectSelectionPolicyState()

        # recall the relevant memories
        description, nretrievals_per_object = format_object_detections(detections_2d, self.lmp_planner.memory_bank, threshold=0)
        print(description, nretrievals_per_object)
        print(self.lmp_planner.memory_bank.memories)

        object_scores = []

        if max(nretrievals_per_object) == 0:
            # we don't remember anything in particular about the objects in the scene
            object_scores = [0] * len(nretrievals_per_object)
        else:
            assert self.lmp_planner is not None
            # have an LLM filtering step for these memories
            prompt = f"Task: {self.lmp_planner.instructions}\n\n"
            if self.lmp_planner.prev_planning is not None:
                prompt += f"""You have written the following plan:
\"""
{self.lmp_planner.prev_planning}
\"""

We are in the middle of the code's execution. The line we are currently executing is
`scene.choose({object_type}, {purpose})`.

This has resulted in the following list of detections:
{description}\n\n"""
            else:
                prompt += f"""
The following objects have been detected:
{description}"""
                
            prompt += """Rank the objects in the scene according to how likely they are to be the best choice.
Respond with three lists: 'likely', 'neutral', and 'unlikely'. Format your list as follows:
```
- likely: [1, 2, 3]
- neutral: [4, 5, 6]
- unlikely: [7, 8, 9]
```
""".strip()
            print("Prompt:")
            print(prompt)

            cmpl = self.lmp_planner.client.chat.completions.create(
                model='gpt-4-turbo',
                messages=[
                    {"role": "system", "content": "You are a helpful human assistant who uses a robot API to help robots make motion plans."},
                    {"role": "user", "content": prompt}
                ]
            )
            message_content = cmpl.choices[0].message.content
            print(message_content)
            object_scores = [int(x) for x in input("type the object scores inferred from above. 1 = likely, 0 = neutral, -1 = unlikely:").split(" ")]

        print("resulting object scores:", object_scores)

        tau = 0.1
        object_scores = np.array(object_scores)
        object_scores = np.exp(object_scores / tau)
        object_scores /= object_scores.sum()
        # choose object
        object_id = np.random.choice(len(object_scores), p=object_scores)

        # now we can try to incorporate human feedback
        annots = draw_set_of_marks(base_rgb_image, detections_2d)
        print(f"Please look at the set of objects. Here, we have selected {object_id+1}.")
        print(f"Is this acceptable? If not, is any other object acceptable instead?")
        # print("However, we would like to know which objects would have been equivalently acceptable or should have been avoided.")
        plt.title("Available objects")
        plt.imshow(annots)
        plt.axis('off')
        plt.show()
        acceptable = 'y' == input("Is this acceptable? (y/n): ")
        if not acceptable:
            negative_object_id = object_id
            object_id = int(input("Which object should be selected instead? (1-indexed): ")) - 1
            if object_id >= 0:
                # now we try to create a memory from this human feedback
                # however we gotta be real that most people will ignore this?
                reason_unacceptable = input("Why was the original choice unacceptable? ")
                # from here we can try to infer some information about that object
                # it's also possible that it's not an intrinsic property of the object,
                # but rather something extrinsic
                prompt = f"""
We are trying to achieve the following task:
{self.lmp_planner.instructions}

You have written the following plan:
\"""
{self.lmp_planner.prev_planning}
\"""

We are in the middle of the code's execution. The line we are currently executing is
`scene.choose({object_type}, {purpose})`.

You originally selected object {object_id + 1} as the best choice. However, the selector has indicated that this object is unacceptable. They have provided the following reason:
{reason_unacceptable}

What relevant information should you remember about object {object_id + 1} in the future? Write a sentence or two.
Write without referring to "object {object_id + 1}" in particular; the object numbering might change in the future.
Imagine that this information will show up alongside the same object class in the future when it appears.
""".strip()
            print("Prompt:")
            print(prompt)

            pos_emb = detections_2d[object_id].embedding_augmented
            neg_emb = detections_2d[negative_object_id].embedding_augmented

            cmpl = self.lmp_planner.client.chat.completions.create(
                model='gpt-4-turbo',
                messages=[
                    {"role": "system", "content": "You are a helpful human assistant who uses a robot API to help robots make motion plans."},
                    {"role": "user", "content": prompt}
                ]
            )
            to_remember = cmpl.choices[0].message.content
            assert to_remember is not None

            print("committing to short-term memory:", to_remember)
            print("positive ID:", object_id)
            print("negative ID:", negative_object_id)

            # now, we commit this to short-term memory
            self.lmp_planner.memory_bank.store(Memory(
                key=MemoryKey(
                    positive_examples=[*pos_emb],
                    negative_examples=[*neg_emb],
                ),
                value=to_remember,
            ))

        detection = detections_2d[object_id]
        x1, y1, x2, y2 = detection.box
        segmented_pcd_xyz, segmented_pcd_color, _segmentation_masks = \
            pcd_segmenter.segment(base_rgb_image, base_point_cloud, [int(x) for x in [x1, y1, x2, y2]], supplementary_rgb_images, supplementary_point_clouds)
        
        return Object(segmented_pcd_xyz, segmented_pcd_color, clip_features=None)

    def choose_v0(self, object_type, purpose):
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

        # filter out detections that are not in the scene
        filtered_detections = []
        for detection in detections_2d:
            xmin, ymin, xmax, ymax = (int(x) for x in detection.box)
            pts = base_point_cloud[ymin:ymax, xmin:xmax]
            valid = ~(pts == -10000).any(axis=-1)
            pts = pts[valid]
            # first, if an object is out of range, there will not be any associated points
            if len(pts) < 10:
                print("filtered out because not enough valid points from point cloud")
                continue
            # then, we can count how many points are in bounds
            pts_flat = pts.reshape(-1, 3)
            in_bounds_pts = (pts_flat[:, 0] > -0.1) & (np.abs(pts_flat[:, 1]) < 3) & (pts_flat[:, 2] > -0.1)
            if (in_bounds_pts.sum() < 10):
                print("filtered out because not enough in-bounds points")
                print(in_bounds_pts)
                print(pts_flat)
                continue
            
            filtered_detections.append(detection)

        detections_2d = filtered_detections
        
        if len(detections_2d) == 0:
            print("No object candidates detected")
            return None

        if self.selection_method_for_choose == 'human':
            im = draw_set_of_marks(image, detections_2d)
            plt.title(f"Please choose: {purpose}")
            plt.imshow(im)
            plt.axis('off')
            plt.show()

            object_id = int(input(f"Choice (choose a number 1 to {len(detections_2d)}, or -1 to cancel): "))

        elif self.selection_method_for_choose == 'vlm':
            im = draw_set_of_marks(image, detections_2d)
            plt.title(f"VLM is choosing: {purpose}")
            plt.imshow(im)
            plt.axis('off')
            plt.show()

            annotated_image = draw_set_of_marks(image, detections_2d)

            object_id = self._choose_with_vlm(annotated_image, object_type, purpose)
        
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
            x1, y1, x2, y2 = [int(x) for x in detection.box]
            segmented_pcd_xyz, segmented_pcd_color, _segmentation_masks = \
                pcd_segmenter.segment(base_rgb_image, base_point_cloud, [x1, y1, x2, y2], supplementary_rgb_images, supplementary_point_clouds)
            
            objects.append(Object(segmented_pcd_xyz, segmented_pcd_color, clip_features=None))

        return objects

class Robot(Panda):
    grasping = False

    def signal_completed(self):
        print("LLM has signaled that the task has been completed")
        input("Operator press enter to continue.")

    def grasp(self, object: Object):
        assert not self.grasping, "Robot is currently in a `grasping` state, and cannot grasp another object. If you believe this is in error, call `robot.release()`."

        # use simplest grasping metric: lowest alpha [surface misalignment] score
        grasps = object.generate_grasps() # each is (alpha, start, end)

        if len(grasps) == 0:
            # huhhh
            print("<warn: no grasps found?>")
            return

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
