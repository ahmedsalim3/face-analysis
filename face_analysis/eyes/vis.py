import cv2
import numpy as np
from .results import EyeStateResultContainer

def draw_bbox(frame: np.ndarray, bbox: np.ndarray, color=(0, 255, 0), thickness=1):

    x_min = int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min = int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max = int(bbox[2])
    y_max = int(bbox[3])

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
    
    return frame

def draw_eye_state(frame: np.ndarray, bbox: np.ndarray, left_state: str, right_state: str, landmarks=None):
    """
    Draw eye state information on frame.
    """
    # Extract safe min and max of x,y
    x_min = int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min = int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max = int(bbox[2])
    y_max = int(bbox[3])
    
    if left_state == "open" and right_state == "open":
        color = (0, 255, 0)
    elif left_state == "closed" and right_state == "closed":
        color = (0, 0, 255)
    else:
        color = (0, 255, 255)
    
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    
    if landmarks is not None and len(landmarks) >= 2:
        left_pos = (int(landmarks[0][0]), int(landmarks[0][1]))
        right_pos = (int(landmarks[1][0]), int(landmarks[1][1]))
    else:
        left_pos = (x_min + int((x_max - x_min) * 0.25), y_min + int((y_max - y_min) * 0.25))
        right_pos = (x_min + int((x_max - x_min) * 0.75), y_min + int((y_max - y_min) * 0.25))
    
    left_color = (0, 255, 0) if left_state == "open" else (0, 0, 255)
    cv2.circle(frame, left_pos, 5, left_color, -1)
    
    right_color = (0, 255, 0) if right_state == "open" else (0, 0, 255)
    cv2.circle(frame, right_pos, 5, right_color, -1)
    
    return frame

def _draw_landmarks(frame: np.ndarray, landmarks: np.ndarray, radius=2, color=(255, 0, 0), thickness=-1):
    """
    Draw facial landmarks on frame
    """
    for point in landmarks:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), radius, color, thickness)
    
    return frame

def render(frame: np.ndarray, results: EyeStateResultContainer, draw_landmarks=False):
    """
    Render eye state detection results on frame
    """

    image_out = frame.copy()
    num_faces = len(results)
    
    if num_faces == 0:
        return image_out
    
    combined_states = results.get_combined_states()
    
    for i in range(num_faces):
        image_out = draw_eye_state(
            image_out, 
            results.bboxes[i], 
            results.left_states[i], 
            results.right_states[i],
            landmarks=results.landmarks[i] if i < len(results.landmarks) else None
        )
        
        if draw_landmarks and i < len(results.landmarks) and results.landmarks[i].shape[0] > 0:
            image_out = _draw_landmarks(image_out, results.landmarks[i])
    
    return image_out
