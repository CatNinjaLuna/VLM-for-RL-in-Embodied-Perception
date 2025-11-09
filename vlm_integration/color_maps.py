# color_maps.py
'''
HSV ranges for CLEVR-like colors used in object segmentation.
'''
import numpy as np
# HSV ranges (0–179, 0–255, 0–255 in OpenCV). Tweak for your renderer.
COLOR_HSV = {
    "red":    [(0,120,70),(10,255,255)],  # plus a second red range:
    "red2":   [(170,120,70),(179,255,255)],
    "green":  [(36, 80, 60),(89,255,255)],
    "blue":   [(90, 80, 60),(130,255,255)],
    "yellow": [(20,120,70),(35,255,255)],
    "magenta":[(140,80,60),(169,255,255)],
    "cyan":   [(80,80,60),(95,255,255)],
    "orange": [(10,120,70),(19,255,255)],
    "purple": [(129,80,60),(155,255,255)]
}

def get_color_masks(hsv, color):
    import cv2
    if color == "red":
        m1 = cv2.inRange(hsv, np.array(COLOR_HSV["red"][0]),  np.array(COLOR_HSV["red"][1]))
        m2 = cv2.inRange(hsv, np.array(COLOR_HSV["red2"][0]), np.array(COLOR_HSV["red2"][1]))
        return m1 | m2
    if color in COLOR_HSV:
        lo, hi = COLOR_HSV[color]
        return cv2.inRange(hsv, np.array(lo), np.array(hi))
    # fallback: no color filter
    return None
