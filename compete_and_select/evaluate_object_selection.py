# Now we can load labels, and see whether OwlV2 is able to detect them.

from dotenv import load_dotenv
load_dotenv()

from .detect_objects import detect
from .sam import boxes_to_masks
from .clip_feature_extraction import get_text_embeds, get_clip_embeddings, get_full_scale_clip_embeddings
from .lmp_scene_api import get_selection_policy

import json
import PIL.Image
import pickle
import os
import torch
from torch.nn.functional import interpolate
import numpy as np
from .standalone_compete_and_select import select_with_vlm, describe_objects

slug_labels = ['blue_mug', 'white_mug', 'green_coffee_cup', 'white_plastic_cup', 'uva_football_cup', 'green_cup']

natural_language_labels = [
    "blue mug",
    "white mug",
    "green coffee cup",
    "white plastic cup",
    "UVA football cup",
    "green plastic cup",
]

folder = './photos/solidbg/cups'

with open(os.path.join(folder, "expected.json")) as f:
    expected_per_img = json.load(f)
    
with open(os.path.join(folder, "masks.pkl"), "rb") as f:
    masks = pickle.load(f)

def obtain_detector_results_folder(folder):
    detector_results_folder = os.path.join(folder, "detector_results")
    if not os.path.exists(detector_results_folder):
        os.makedirs(detector_results_folder)
    return detector_results_folder

def generate_clip_embeddings():
    clip_embeddings = {}
    
    for filename in sorted(expected_per_img.keys()):
        print("Opening", filename, "...")
        
        full_filename = os.path.join(folder, filename)
        image = PIL.Image.open(full_filename).convert("RGB")

        clip_embeddings[filename] = get_full_scale_clip_embeddings(image)
        
    detector_results_folder = obtain_detector_results_folder(folder)
    with open(os.path.join(detector_results_folder, "clip_embeddings.pkl"), "wb") as f:
        pickle.dump(clip_embeddings, f)

def generate_direct_owlv2():
    # store in raw python format
    detections = {}

    for filename in sorted(expected_per_img.keys()):
        print("Opening", filename, "...")
        full_filename = os.path.join(folder, filename)
        image = PIL.Image.open(full_filename)
        
        detections_for_file = []
        labels_in_this_file = expected_per_img[filename]
        masks_in_this_file = masks[filename]
        gt_mask_map = {
            label: mask for (label, mask) in zip(labels_in_this_file, masks_in_this_file)
        }

        # go through natural language labels
        for k, label in enumerate(natural_language_labels):
            dets = detect(image, label, use_clip_projection=True)

            print(f"Detecting [{label}]: # = {len(dets)}")

            masks_for_detection_set = boxes_to_masks(image, [det.box for det in dets])

            detections_for_file.append({
                "detections": [
                    {
                        "bbox": det.box,
                        "score": det.score,
                        "clip_embed_rescaled_cropped": det.embedding,
                        "mask": masks_for_detection_set[i]
                    }
                    for i, det in enumerate(dets)
                ],
                "natural_language_label": label,
                "label": slug_labels[k],
            })
            
        detections[filename] = detections_for_file

    detector_results_folder = obtain_detector_results_folder(folder)
    with open(os.path.join(detector_results_folder, "owlv2_direct.pkl"), "wb") as f:
        pickle.dump(detections, f)

def mask_iou(m1, m2):
    return (m1 & m2).sum() / (m1 | m2).sum()

# cache the results to preserve credits.
def generate_vlm_results__(image, detections_for_file, allow_descriptions=True):
    vlm_outputs = []
    
    for detection_group in detections_for_file:
        print("Running on detection group:", detection_group['label'])
        bboxes = [det['bbox'] for det in detection_group['detections']]
        
        if allow_descriptions:
            descriptions = describe_objects(image, bounding_boxes)
        else:
            descriptions = None
        
        # has "reasoning", "logits", and "response" keys
        # returns a list of logits, corresponding to bboxes.
        vlm_output = select_with_vlm(image, bboxes, detection_group['natural_language_label'], descriptions)
        vlm_outputs.append(vlm_output)
        
    return vlm_outputs

def generate_vlm_results(image, detections_for_file, allow_descriptions=True):
    from concurrent.futures import ThreadPoolExecutor
    
    def inner(detection_group):
        print("Running on detection group:", detection_group['label'])
        bboxes = [det['bbox'] for det in detection_group['detections']]
        
        if allow_descriptions:
            descriptions = describe_objects(image, bounding_boxes)
        else:
            descriptions = None
        
        # has "reasoning", "logits", and "response" keys
        # returns a list of logits, corresponding to bboxes.
        return select_with_vlm(image, bboxes, detection_group['natural_language_label'], descriptions)
    
    with ThreadPoolExecutor() as ex:
        return list(ex.map(inner, detections_for_file))

def obtain_vlm_results(allow_descriptions):
    # may use the cache. REQUIRES: owlv2_direct.pkl
    detector_results_folder = obtain_detector_results_folder(folder)
    out_path = os.path.join(detector_results_folder, f"owlv2_direct__vlm_outputs__{'yes' if allow_descriptions else 'no'}_descriptions.pkl")
    
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            return pickle.load(f)
        
    with open(os.path.join(detector_results_folder, "owlv2_direct.pkl"), "rb") as f:
        detections = pickle.load(f)
        
    generations = {}
        
    for filename in sorted(expected_per_img.keys()):
        print("Opening", filename, "...")
        full_filename = os.path.join(folder, filename)
        image = PIL.Image.open(full_filename).convert("RGB")
        detections_for_file = detections[filename]
        generations[filename] = generate_vlm_results(image, detections_for_file, allow_descriptions)
        
    with open(out_path, "wb") as f:
        pickle.dump(generations, f)
        
    return generations
        

def evaluate_detection_results():
    print("Beginning to evaluate detection results (OwlV2_Direct)")
    
    with open(os.path.join(folder, "detector_results/owlv2_direct.pkl"), "rb") as f:
        detections = pickle.load(f)
        
    with open(os.path.join(folder, "detector_results/owlv2_direct__vlm_outputs__yes_descriptions.pkl"), "rb") as f:
        vlm_results_yes_descriptions = pickle.load(f)
      
    # This is a lot of I/O for not a lot of compute
    # with open(os.path.join(folder, "detector_results/clip_embeddings.pkl"), "rb") as f:
    #     clip_embeddings = pickle.load(f)
    
    # now take whatever detection had the highest score and use that as the prediction.
    # we then match to the ground truth mask to see which object was actually highlighted.
    score_argmax_results = {
        'folder': [],
        'filename': [],
        'expected_label': [],
        'iou_owlv2_score': [],
        'iou_clip_cropped_score': [],
    }
    
    text_embeds = get_text_embeds(["a photo of a " + label for label in natural_language_labels])
    text_embeds = {slug_label: embed for (slug_label, embed) in zip(slug_labels, text_embeds)}
    
    for filename in sorted(detections.keys()):
        print("Opening", filename)
        
        full_filename = os.path.join(folder, filename)
        image = PIL.Image.open(full_filename).convert("RGB")
        clip_embeds = get_full_scale_clip_embeddings(image)['rescaled_cropped']
        expected_for_file = expected_per_img[filename]
        # this contains all detections. not necessarily in the order of expected_for_file.
        detections_for_file = detections[filename]
        
        vlm_results_yes_descriptions_for_file = vlm_results_yes_descriptions[filename]
        
        for expected_label, target_mask in zip(expected_for_file, masks[filename]):
            text_embed = text_embeds[expected_label]
            
            print(expected_label)
            
            # check and see what the result was when we tried to detect this label
            index_in_detections = [i for i in range(len(detections_for_file)) if detections_for_file[i]['label'] == expected_label][0]
            dets = detections_for_file[index_in_detections]['detections']
            
            vlm_pred_index = np.argmax(vlm_results_yes_descriptions_for_file[index_in_detections]['logits'])
            best_det_vlm_pred = dets[vlm_pred_index]
            iou_vlm = mask_iou(best_det_vlm_pred['mask'], target_mask)
            
            # take the score argmax
            best_det_owlv2_score = max(dets, key=lambda det: det['score'])
            iou_owlv2 = mask_iou(best_det_owlv2_score['mask'], target_mask)
            
            best_det_clip_cropped_score = max(dets, key=lambda det: det['clip_embed_rescaled_cropped'] @ text_embed)
            iou_clip_cropped = mask_iou(best_det_clip_cropped_score['mask'], target_mask)
            
            # take the CLIP mapped embedding
            best_det_clip_mapped_score = max(dets, key=lambda det: clip_embeds[det['mask']].mean(axis=0) @ text_embed)
            iou_clip_mapped = mask_iou(best_det_clip_mapped_score['mask'], target_mask)
            
            print(f"IOU[vlm]: {iou_vlm:.2f}, IOU [owlv2]: {iou_owlv2:.2f}, IOU [clip_cropped]: {iou_clip_cropped:.2f}, IOU [clip_mapped]: {iou_clip_mapped:.2f}")
        
# generate_direct_owlv2()
# generate_clip_embeddings()
# obtain_vlm_results(allow_descriptions=True)
evaluate_detection_results()
