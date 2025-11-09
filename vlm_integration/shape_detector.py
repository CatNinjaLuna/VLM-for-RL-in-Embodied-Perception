# shape_detector.py
'''
circles, squares/rects via contours and Hough transform.
'''
import cv2
import numpy as np

def find_candidates(bgr, mask=None):
    """Returns list of dicts: {contour, bbox, centroid, shape}"""
    img = bgr.copy()
    if mask is not None:
        img = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cand = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50: 
            continue
        x,y,w,h = cv2.boundingRect(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        M = cv2.moments(c)
        if M["m00"] == 0: 
            continue
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

        # classify shape
        shape = "unknown"
        if len(approx) >= 8:
            # try Hough circle on cropped region
            roi = gray[y:y+h, x:x+w]
            circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(w,h)//3,
                                       param1=100, param2=12, minRadius=5, maxRadius=max(5,min(w,h)//2))
            if circles is not None:
                shape = "circle"
        if shape == "unknown":
            if len(approx) == 4:
                ar = w / float(h)
                shape = "square" if 0.85 <= ar <= 1.15 else "rectangle"

        cand.append({
            "contour": c, "bbox": (x,y,w,h), "centroid": (cx,cy), "shape": shape, "area": area
        })
    return cand
