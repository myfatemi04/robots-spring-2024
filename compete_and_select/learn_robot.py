from dataclasses import dataclass
from functools import cached_property, partial
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openai
import PIL.Image
import PIL.ImageFilter
from detect_objects import detect, embed_box, get_clip_embeddings
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from memory_bank_v2 import MemoryBank
from select_object_v2 import draw_set_of_marks
from sklearn.svm import SVC


def select_bounding_box(image):
    plt.rcParams['figure.figsize'] = (20, 10)

    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        # print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        # print(" The button you used were: %s %s" % (eclick.button, erelease.button))

    def toggle_selector(event):
        # print(' Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            # print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            # print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)


    fig, current_ax = plt.subplots()
    # print("\n      click  -->  release")

    # drawtype is 'box' or 'line' or 'none'
    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                        useblit=True,
                                        button=[1, 3], # type: ignore
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.imshow(image)
    plt.show()

    # get value
    x1, x2, y1, y2 = toggle_selector.RS.extents
    return int(x1), int(y1), int(x2), int(y2)

def teach_robot():
    memory_bank = MemoryBank()

    # Here, the human provides some direct annotations for what to do (e.g. circling an object and saying what to do with it)
    # Human gives an instruction.
    # Robot asks "OK, could you walk me through it?" or something similar.
    user = "Michael"
    instructions = "Put the snacks in the bowl"
    image = PIL.Image.open("sample_images/IMG_8650.jpeg")

    # detections = detect(image, "snack bag")
    # print(detections)
    # drawn = draw_set_of_marks(image, detections)
    # plt.imshow(drawn)
    # plt.axis('off')
    # plt.show()

    scene_key, embedding_map = get_clip_embeddings(image)
    scene_key_2, embedding_map_2 = get_clip_embeddings(image.filter(PIL.ImageFilter.GaussianBlur(2)))

    box = select_bounding_box(image)
    embeds_inside_box = []
    embeds_outside_box = []

    for embed_grid in [embedding_map, embedding_map_2]:
        for row in range(16):
            for col in range(16):
                x = col * 14 * (image.width / 224)
                y = row * 14 * (image.height / 224)
                if (box[0] <= x <= box[2]) and (box[1] <= y <= box[3]):
                    embeds_inside_box.append(embed_grid[row, col])
                else:
                    embeds_outside_box.append(embed_grid[row, col])

    embeds = np.stack(embeds_inside_box + embeds_outside_box)
    class_labels = [1]*len(embeds_inside_box) + [0]*len(embeds_outside_box)
    # sample_weights = [1]*len(embeds_inside_box) + [len(embeds_inside_box)/len(embeds_outside_box)]*len(embeds_outside_box)
    svm = SVC(probability=True)
    svm = svm.fit(embeds, class_labels) # , sample_weights)

    # NEW IMAGE TIME!!!
    image = PIL.Image.open("sample_images/IMG_8651.jpeg")
    embed_grid = get_clip_embeddings(image)[1]

    grid_match_score = svm.predict_proba(embed_grid.reshape(-1, 1024))[:, 1].reshape(16, 16)

    print(grid_match_score)

    # grid_match_score = (embed_grid * box_embed).sum(axis=-1)/(np.linalg.norm(embed_grid, axis=-1)*np.linalg.norm(box_embed))
    # grid_match_score = grid_match_score - grid_match_score.min()
    # grid_match_score = grid_match_score / grid_match_score.max()

    match_image = PIL.Image.fromarray(np.uint8(grid_match_score * 255)).resize((image.width, image.height))
    match_image = np.array(match_image) / 255

    plt.imshow(image)
    plt.imshow(match_image, alpha=match_image)

    # plt.imshow(match_image)
    plt.axis('off')
    plt.show()

def in_box(box, x, y):
    return (box[0] <= x <= box[2]) and (box[1] <= y <= box[3])

@dataclass
class ImageObservation:
    image: PIL.Image.Image
    @cached_property
    def embeddings(self):
        return get_clip_embeddings(self.image)
    @property
    def embedding(self):
        return self.embeddings[0]
    @property
    def embedding_map(self):
        return self.embeddings[1]

def get_embeddings_in_positive_boxes(obs: ImageObservation, positive_boxes):
    positive_embeddings = []
    negative_embeddings = []

    for row in range(16):
        for col in range(16):
            x = (col + 0.5) * 14 * (obs.image.width / 224)
            y = (row + 0.5) * 14 * (obs.image.height / 224)

            target_list = positive_embeddings if any(in_box(box, x, y) for box in positive_boxes) else negative_embeddings
            target_list.append(obs.embedding_map[row, col])

    return positive_embeddings, negative_embeddings

MIN_OCCUPANCY = 0.1

def get_include_mask(obs: ImageObservation, mask: np.ndarray):
    include_mask = np.zeros((16, 16), dtype=bool)
    for row in range(16):
        for col in range(16):
            start_x = int(col * 14 * (obs.image.width / 224))
            start_y = int(row * 14 * (obs.image.height / 224))
            end_x = int((col + 1) * 14 * (obs.image.width / 224))
            end_y = int((row + 1) * 14 * (obs.image.height / 224))
            mask_occupancy = np.mean(mask[start_y:end_y, start_x:end_x])

            if mask_occupancy > MIN_OCCUPANCY:
                include_mask[row, col] = True
    return include_mask

def get_embeddings_under_mask(obs: ImageObservation, mask: np.ndarray):
    positive_embeddings = []
    negative_embeddings = []

    for row in range(16):
        for col in range(16):
            start_x = int(col * 14 * (obs.image.width / 224))
            start_y = int(row * 14 * (obs.image.height / 224))
            end_x = int((col + 1) * 14 * (obs.image.width / 224))
            end_y = int((row + 1) * 14 * (obs.image.height / 224))
            mask_occupancy = np.mean(mask[start_y:end_y, start_x:end_x])

            target_list = positive_embeddings if mask_occupancy > MIN_OCCUPANCY else negative_embeddings
            target_list.append(obs.embedding_map[row, col])

    return positive_embeddings, negative_embeddings

# takes positive examples for a set of observations and compiles them into an indicator function
def compile_memories_with_boxes(images: List[ImageObservation], positive_boxes_per_image):
    positive_embeddings = []
    negative_embeddings = []

    for obs, positive_boxes in zip(images, positive_boxes_per_image):
        pos, neg = get_embeddings_in_positive_boxes(obs, positive_boxes)
        positive_embeddings.extend(pos)
        negative_embeddings.extend(neg)

    embeds = np.stack(positive_embeddings + negative_embeddings)
    class_labels = [1]*len(positive_embeddings) + [0]*len(negative_embeddings)
    svm = SVC(probability=True)
    svm = svm.fit(embeds, class_labels)

    return svm

def compile_memories(images: List[ImageObservation], masks_per_image):
    positive_embeddings = []
    negative_embeddings = []

    for obs, mask in zip(images, masks_per_image):
        pos, neg = get_embeddings_under_mask(obs, mask)
        positive_embeddings.extend(pos)
        negative_embeddings.extend(neg)

    print(len(positive_embeddings), len(negative_embeddings))

    embeds = np.stack(positive_embeddings + negative_embeddings)
    class_labels = [1]*len(positive_embeddings) + [0]*len(negative_embeddings)
    svm = SVC(probability=True)
    svm = svm.fit(embeds, class_labels)

    return svm

def highlight(image: ImageObservation, svm: SVC):
    embed_grid = image.embedding_map
    grid_match_score = svm.predict_proba(embed_grid.reshape(-1, 1024))[:, 1].reshape(16, 16)
    return grid_match_score

def visualize_highlight(image: ImageObservation, grid_match_score):
    match_image = PIL.Image.fromarray(np.uint8(grid_match_score * 255)).resize((image.image.width, image.image.height))
    match_image = np.array(match_image) / 255

    plt.imshow(image.image)
    plt.imshow(match_image, alpha=match_image)

    plt.axis('off')
    plt.show()

import torch
from transformers import SamModel, SamProcessor

# to create a better mask, we'll use the bounding boxes as prompts for SAM, which can try to exclude some of the background
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model: SamModel = SamModel.from_pretrained("facebook/sam-vit-base").to(device) # type: ignore
sam_processor: SamProcessor = SamProcessor.from_pretrained("facebook/sam-vit-base") # type: ignore

def boxes_to_masks(image: PIL.Image.Image, input_boxes: List[Tuple[int, int, int, int]]):
    inputs = sam_processor(images=[image], input_boxes=[[list(box) for box in input_boxes]], return_tensors="pt").to(device)
    outputs = sam_model(**inputs)
    masks = sam_processor.image_processor.post_process_masks( # type: ignore
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),      # type: ignore
        inputs["reshaped_input_sizes"].cpu() # type: ignore
    )[0] # 0 to remove batch dimension
    return masks

# we now create an artificial narration for the robot to learn from
def artificial_narration():
    steps = [
        "I'm going to show you how to clean these items up. Food items should go in the cardboard box and rags should go in the bowl.",
        PIL.Image.open("sample_images/cleanup_sequence/IMG_8655.jpeg"),
        "Put {object1} into {object2}. This is a food item.",
        PIL.Image.open("sample_images/cleanup_sequence/IMG_8658.jpeg"),
    ]

    train_image = ImageObservation(steps[1])
    train_image_aug = ImageObservation(steps[1].filter(PIL.ImageFilter.GaussianBlur(2)))
    # very_easy_eval_image = train_image
    eval_image = PIL.Image.open("sample_images/cleanup_sequence/IMG_8654.jpeg")
    very_easy_eval_image = ImageObservation(eval_image)
    brown_oatmeal = (1376, 1967, 2217, 2638)
    cardboard_box = (275, 788, 1298, 1533)
    blue_rag = (1964, 1698, 2602, 2177)
    red_oatmeal = (2394, 1519, 2957, 1982)
    white_rag = (2924, 1603, 3652, 2120)
    teddy_grahams = (2262, 2132, 3099, 2642)
    bowl = (1435, 645, 2431, 1468)

    food_item_train_examples = [
        brown_oatmeal,
        red_oatmeal,
        teddy_grahams,
    ]
    rag_train_examples = [
        blue_rag,
        white_rag,
    ]
    bowl_train_examples = [bowl]
    box_train_examples = [cardboard_box]

    # visualize the masks more accurately
    overall_mask = np.zeros((train_image.image.height, train_image.image.width), dtype=bool)
    masks = boxes_to_masks(train_image.image, food_item_train_examples)

    for mask in masks:
        mask = mask[0].detach().cpu().numpy().astype(float)
        overall_mask |= (mask > 0.5)

    # center crop.
    offset = (train_image.image.width - train_image.image.height) // 2
    overall_mask = overall_mask[:, offset:-offset]
    train_image.image = train_image.image.crop((offset, 0, train_image.image.height + offset, train_image.image.height))
    train_image_aug.image = train_image_aug.image.crop((offset, 0, train_image_aug.image.height + offset, train_image_aug.image.height))

    plt.imshow(train_image.image)
    plt.imshow(overall_mask.astype(float), alpha=overall_mask.astype(float))
    plt.axis('off')
    plt.show()

    visualize_highlight(train_image, get_include_mask(train_image, overall_mask))

    food_item_indicator_function = compile_memories([train_image, train_image_aug], [overall_mask, overall_mask])

    # center crop.
    very_easy_eval_image.image = very_easy_eval_image.image.crop((offset, 0, very_easy_eval_image.image.height + offset, very_easy_eval_image.image.height))

    visualize_highlight(
        very_easy_eval_image,
        highlight(very_easy_eval_image, food_item_indicator_function)
    )

    return

    # We can do a forward pass, measure error, and backpropagate to learn the policy
    # During backpropagation, we should see what reasoning steps led to the choice, and see what needs to change about the reasoning step and/or the background knowledge
    # We can prompt engineer to prevent overfitting
    # Should provide a list of available memory classes (e.g. "food", "bowl", "box", "rag"), treat as an online object detection problem, where we add tags to certain objects

    # When the robot sees the same situation, it should be reminded of its past experience cleaning it up
    # Incorporate increased selectivity *later*.

    # In this case we instruct the robot to perform some task with the objects
    # And then the robot should learn facts about object1 and object2 (e.g. object1 is an example of something that should go in object2)
    # Ideally the robot generates all the information necessary to perform the policy accurately

    # robot should generate a piece of information or set of facts related to object 1
    client = openai.OpenAI()
    chat = partial(client.chat.completions.create, model='gpt-4-turbo-preview')
    content = chat(messages=[
        {'role': 'system', 'content': "You are a system which generates useful information based on interactions you observe between a human and a robot."},
        {'role': 'user', 'content': f"{steps[0]}\n<Image>\n{steps[2]}\n<Image>"},
        {'role': 'user', 'content': \
         """Based on how they were used and referred to in this interaction, what can you say about {object1} and {object2}?"""},
    ]).choices[0].message.content
    print(content)


# teach_robot()
artificial_narration()
