# grounder.py
'''
Ground textual instructions to detected objects in an image.
'''
import cv2
import numpy as np
from text_parser import parse_instruction
from color_maps import get_color_masks
from shape_detector import find_candidates
from spatial_reasoner import select_by_relation

def ground_instruction(bgr_image, instruction):
    parsed = parse_instruction(instruction)
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Detect candidates for main target (by color+shape)
    target_mask = get_color_masks(hsv, parsed["color"]) if parsed["color"] else None
    t_cand = find_candidates(bgr_image, target_mask)
    if parsed["shape"]:
        t_cand = [c for c in t_cand if c["shape"] == parsed["shape"]]

    # If relation exists, detect reference set and pick target by relation
    if parsed["relation"]:
        ref_mask = get_color_masks(hsv, parsed["ref_color"]) if parsed["ref_color"] else None
        r_cand = find_candidates(bgr_image, ref_mask)
        if parsed["ref_shape"]:
            r_cand = [c for c in r_cand if c["shape"] == parsed["ref_shape"]]
        chosen = select_by_relation(t_cand, r_cand, parsed["relation"])
    else:
        # choose largest area if multiple
        chosen = max(t_cand, key=lambda x: x["area"]) if t_cand else None

    return parsed, chosen
