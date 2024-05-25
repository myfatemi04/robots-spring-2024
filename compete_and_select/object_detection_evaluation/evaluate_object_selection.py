# Now we can load labels, and see whether OwlV2 is able to detect them.

from dotenv import load_dotenv

load_dotenv()

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from torch.nn.functional import interpolate

from .class_labels import class_labels
from ..clip_feature_extraction import (get_clip_embeddings,
                                       get_full_scale_clip_embeddings,
                                       get_text_embeds)
from ..detect_objects import detect
from ..lmp_scene_api import get_selection_policy
from ..sam import boxes_to_masks
from .standalone_compete_and_select import describe_objects, select_with_vlm


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

def generate_direct_owlv2(coarse_label=None):
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
            dets = detect(image, coarse_label or label, use_clip_projection=True)
            # dets = detect(image, label, use_clip_projection=True)

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
def generate_vlm_results(image, detections_for_file, allow_descriptions, used_coarse_detections, dry_run):
    from concurrent.futures import ThreadPoolExecutor
    
    if dry_run:
        print("==> Dry Run <== (Only for debugging, not making LM calls)")
    
    if used_coarse_detections and allow_descriptions and not dry_run:
        bboxes = [d['bbox'] for d in detections_for_file[0]['detections']]
        coarse_descriptions = describe_objects(image, bboxes)
    else:
        coarse_descriptions = None
    
    def inner(detection_group):
        print("Running on detection group:", detection_group['label'])
        bboxes = [det['bbox'] for det in detection_group['detections']]
        
        if allow_descriptions:
            if coarse_descriptions:
                # reuse coarse descriptions
                descriptions = coarse_descriptions
            elif not dry_run:
                descriptions = describe_objects(image, bboxes)
        else:
            descriptions = None
            
        print("Allow Descriptions:", descriptions is not None)
        
        # has "reasoning", "logits", and "response" keys
        # returns a list of logits, corresponding to bboxes.
        return select_with_vlm(image, bboxes, detection_group['natural_language_label'], descriptions, dry_run=dry_run)
    
    with ThreadPoolExecutor() as ex:
        return list(ex.map(inner, detections_for_file))

def obtain_vlm_results(allow_descriptions, used_coarse_detections, dry_run=False):
    # may use the cache. REQUIRES: owlv2_direct.pkl
    detector_results_folder = obtain_detector_results_folder(folder)
    out_path = os.path.join(detector_results_folder, f"owlv2_direct__vlm_outputs__{'yes' if allow_descriptions else 'no'}_descriptions.pkl")
    
    if not dry_run:
        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                print("[cache hit: VLM results]")
                return pickle.load(f)
        
    with open(os.path.join(detector_results_folder, "owlv2_direct.pkl"), "rb") as f:
        detections = pickle.load(f)
        
    generations = {}
        
    for filename in sorted(expected_per_img.keys()):
        print("Opening", filename, "...")
        full_filename = os.path.join(folder, filename)
        image = PIL.Image.open(full_filename).convert("RGB")
        detections_for_file = detections[filename]
        generations[filename] = generate_vlm_results(image, detections_for_file, allow_descriptions, used_coarse_detections, dry_run)
        
    if not dry_run:
        with open(out_path, "wb") as f:
            pickle.dump(generations, f)
        
    return generations
        

def evaluate_detection_results():
    print("Beginning to evaluate detection results (OwlV2_Direct)")
    
    with open(os.path.join(folder, "detector_results/owlv2_direct.pkl"), "rb") as f:
        detections = pickle.load(f)
        
    with open(os.path.join(folder, "detector_results/owlv2_direct__vlm_outputs__yes_descriptions.pkl"), "rb") as f:
        vlm_results_yes_descriptions = pickle.load(f)
    with open(os.path.join(folder, "detector_results/owlv2_direct__vlm_outputs__no_descriptions.pkl"), "rb") as f:
        vlm_results_no_descriptions = pickle.load(f)
      
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
    
    text_embeds = get_text_embeds(["a photo of " + label for label in natural_language_labels])
    text_embeds = {slug_label: embed for (slug_label, embed) in zip(slug_labels, text_embeds)}
    
    print(text_embeds.keys())
    
    counter = 0
    # os.mkdir('plots', exist_ok=True)
    
    for filename in sorted(detections.keys()):
        print("Opening", filename)
        
        full_filename = os.path.join(folder, filename)
        image = PIL.Image.open(full_filename).convert("RGB")
        clip_embeds = get_full_scale_clip_embeddings(image)['rescaled_cropped']
        expected_for_file = expected_per_img[filename]
        # this contains all detections. not necessarily in the order of expected_for_file.
        detections_for_file = detections[filename]
        
        vlm_results_yes_descriptions_for_file = vlm_results_yes_descriptions[filename]
        vlm_results_no_descriptions_for_file = vlm_results_no_descriptions[filename]
        
        for expected_label, target_mask in zip(expected_for_file, masks[filename]):
            text_embed = text_embeds[expected_label]
            
            print(expected_label)
            
            # check and see what the result was when we tried to detect this label
            index_in_detections = [i for i in range(len(detections_for_file)) if detections_for_file[i]['label'] == expected_label][0]
            dets = detections_for_file[index_in_detections]['detections']
            
            vlm_yes_pred_index = np.argmax(vlm_results_yes_descriptions_for_file[index_in_detections]['logits'])
            best_det_vlm_yes_pred = dets[vlm_yes_pred_index]
            iou_vlm_yes = mask_iou(best_det_vlm_yes_pred['mask'], target_mask)
            
            vlm_no_pred_index = np.argmax(vlm_results_no_descriptions_for_file[index_in_detections]['logits'])
            best_det_vlm_no_pred = dets[vlm_no_pred_index]
            iou_vlm_no = mask_iou(best_det_vlm_no_pred['mask'], target_mask)
            
            # draw the vlm pred mask
            counter += 1
            # plt.clf()
            # plt.title("Result: " + filename + " => " + expected_label)
            # plt.imshow(image)
            # plt.imshow(target_mask.astype(float), alpha=target_mask.astype(float))
            # # plt.imshow(best_det_vlm_pred['mask'].astype(float), alpha=best_det_vlm_pred['mask'].astype(float))
            # plt.savefig(f'plots/plot_{counter}.png')
            
            # take the score argmax
            best_det_owlv2_score = max(dets, key=lambda det: det['score'])
            iou_owlv2 = mask_iou(best_det_owlv2_score['mask'], target_mask)
            
            best_det_clip_cropped_score = max(dets, key=lambda det: det['clip_embed_rescaled_cropped'] @ text_embed)
            iou_clip_cropped = mask_iou(best_det_clip_cropped_score['mask'], target_mask)
            
            # take the CLIP mapped embedding
            best_det_clip_mapped_score = max(dets, key=lambda det: clip_embeds[det['mask']].mean(axis=0) @ text_embed)
            iou_clip_mapped = mask_iou(best_det_clip_mapped_score['mask'], target_mask)
            
            print(f"IOU[c+s]: {iou_vlm_yes:.2f}, IOU[som]: {iou_vlm_no:.2f}, IOU [owlv2]: {iou_owlv2:.2f}, IOU [clip_cropped]: {iou_clip_cropped:.2f}, IOU [clip_mapped]: {iou_clip_mapped:.2f}")
            
gp = 'cups'
natural_language_labels = class_labels[gp]['natural_language_labels']
slug_labels = class_labels[gp]['slug_labels']

folder = './photos/solidbg/' + gp

with open(os.path.join(folder, "expected.json")) as f:
    expected_per_img = json.load(f)
    
with open(os.path.join(folder, "masks.pkl"), "rb") as f:
    masks = pickle.load(f)

# generate_direct_owlv2(coarse_label='spoon')
obtain_vlm_results(allow_descriptions=True, used_coarse_detections=True, dry_run=False)
obtain_vlm_results(allow_descriptions=False, used_coarse_detections=True, dry_run=False)
evaluate_detection_results()
