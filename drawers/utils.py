import cv2 
import numpy as np
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

def draw_traingle(frame,bbox,color):
    """
    Returns:
        numpy.ndarray: The frame with the triangle drawn on it.
    """
    y= int(bbox[1])
    x,_ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x,y],
        [x-10,y-20],
        [x+10,y-20],
    ])
    cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

    return frame

def draw_ellipse(frame, bbox, color, track_id=None):
    # 1. Extract the bounding box coordinates and ensure they are integers
    x1, y1, x2, y2 = map(int, bbox)
    
    # 2. Draw the main rectangle around the player
    cv2.rectangle(
        frame,
        (x1, y1),  # Top-left point
        (x2, y2),  # Bottom-right point
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    # 3. Handle the Track ID Box (Centered at the bottom of the player)
    if track_id is not None:
        x_center = (x1 + x2) // 2
        rect_width, rect_height = 40, 20
        
        # Position the ID box slightly below the player's feet (y2)
        rx1 = x_center - rect_width // 2
        ry1 = y2 + 5
        rx2 = x_center + rect_width // 2
        ry2 = y2 + 5 + rect_height

        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0,0,0), cv2.FILLED)
        
        # Adjust text position based on ID length
        text_offset = 12 if track_id < 100 else 2
        cv2.putText(
            frame,
            f"{track_id}",
            (rx1 + text_offset, ry1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    return frame