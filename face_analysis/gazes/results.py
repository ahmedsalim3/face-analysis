from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class GazeResultContainer:

    pitch: np.ndarray
    yaw: np.ndarray
    bboxes: np.ndarray
    landmarks: np.ndarray
    scores: np.ndarray
    looking_at_camera: bool = False

    def __post_init__(self):
        self.looking_at_camera = self.is_looking_at_camera()
    
    def is_looking_at_camera(self, threshold=15):
        pitch_degrees = np.degrees(self.pitch)
        yaw_degrees = np.degrees(self.yaw)
        
        return (abs(pitch_degrees) < threshold) & (abs(yaw_degrees) < threshold)
    
    def to_dataframe(self):
        data = {
            'face_bboxes': [bbox if isinstance(bbox, (list, tuple)) else [bbox] for bbox in self.bboxes],
            'landmarks': [landmark if isinstance(landmark, (list, tuple)) else [landmark] for landmark in self.landmarks],
            'looking_at_camera': [camera for camera in self.looking_at_camera],
            'pitch': [pitch for pitch in self.pitch],
            'scores': [score for score in self.scores],
            'yaw': [yaw for yaw in self.yaw]
        }

        return pd.DataFrame(data)

# def is_looking_at_camera(pitch, yaw, threshold=15):
#     if abs(pitch) < np.deg2rad(threshold) and abs(yaw) < np.deg2rad(threshold):
#         return True
#     return False