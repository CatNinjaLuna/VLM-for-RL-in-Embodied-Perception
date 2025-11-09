# demo_grounding.py
'''
Demonstration of grounding textual instructions to objects in an image.'''
import cv2
from ground_instruction import ground_instruction

INSTR = "go to the blue sphere left of the red cube"  # try variants
IMG = "sample_frame.png"                               # put a CLEVR-like frame here

bgr = cv2.imread(IMG)
parsed, det = ground_instruction(bgr, INSTR)

vis = bgr.copy()
if det:
    x,y,w,h = det["bbox"]
    cx,cy = det["centroid"]
    cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.circle(vis,(cx,cy),4,(0,255,0),-1)
    cv2.putText(vis, f"{det['shape']}", (x,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
else:
    cv2.putText(vis, "NO MATCH", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255),2)

cv2.imwrite("grounded_vis.png", vis)
print("Parsed:", parsed)
print("Detection:", det)
print("Visualization saved to grounded_vis.png")