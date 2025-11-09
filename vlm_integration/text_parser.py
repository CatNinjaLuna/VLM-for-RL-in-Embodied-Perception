# text_parser.py
'''
Extract object and spatial relation information from textual instructions.
'''
import re
COLORS = ["red","green","blue","yellow","cyan","magenta","purple","gray","black","white","brown","orange"]
SHAPES = ["sphere","ball","circle","cube","box","square","cylinder","rectangle"]

REL_WORDS = {
    "left_of": ["left of","to the left of"],
    "right_of": ["right of","to the right of"],
    "in_front_of": ["in front of"],
    "behind": ["behind"]
}

def parse_instruction(text: str):
    t = text.lower().strip()
    color = next((c for c in COLORS if c in t), None)
    # map synonyms
    shape_tok = next((s for s in SHAPES if s in t), None)
    if shape_tok in ("sphere","ball","circle"): shape = "circle"
    elif shape_tok in ("cube","box","square"):  shape = "square"
    elif shape_tok in ("cylinder","rectangle"): shape = "rectangle"
    else: shape = None

    relation, ref_color, ref_shape = None, None, None
    for key, phrases in REL_WORDS.items():
        for p in phrases:
            if p in t:
                relation = key
                # try to capture "X left of Y"
                # crude: after phrase, look for next color/shape
                after = t.split(p,1)[1]
                ref_color = next((c for c in COLORS if c in after), None)
                ref_shape_tok = next((s for s in SHAPES if s in after), None)
                if ref_shape_tok in ("sphere","ball","circle"): ref_shape = "circle"
                elif ref_shape_tok in ("cube","box","square"):  ref_shape = "square"
                elif ref_shape_tok in ("cylinder","rectangle"): ref_shape = "rectangle"
                break
        if relation: break

    return {
        "color": color, "shape": shape,
        "relation": relation, "ref_color": ref_color, "ref_shape": ref_shape
    }
