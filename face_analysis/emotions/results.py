from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd

@dataclass
class EmotionResultContainer:
    """
    Container for emotion detection results.
    """
    boxes: np.ndarray
    emotions: List[Dict[str, float]]
    scores: np.ndarray
    
    def __len__(self):
        """Return the number of faces detected."""
        return len(self.boxes)

    def get_top_emotions(self):
        """Get the top emotion for each face."""
        if not self.emotions:
            return []
        return [max(emotion_dict, key=emotion_dict.get) for emotion_dict in self.emotions]
    
    def filter_by_confidence(self, threshold: float = 0.5):
        """
        Filter results by confidence score
        """
        mask = self.scores >= threshold
        return EmotionResultContainer(
            boxes=self.boxes[mask],
            emotions=[emotion for i, emotion in enumerate(self.emotions) if mask[i]],
            scores=self.scores[mask]
        )

    def get_face_details(self, face_idx: int = 0):
        """
        Get detailed information for a specific face
        """
        if face_idx >= len(self.boxes) or face_idx < 0:
            return None

        top_emotion = max(self.emotions[face_idx], key=self.emotions[face_idx].get)

        return {
            "box": self.boxes[face_idx],
            "emotions": self.emotions[face_idx],
            "top_emotion": top_emotion,
            "confidence": self.emotions[face_idx][top_emotion],
        }
    
    def to_dataframe(self):
        data = {
            'face_bboxes': [bbox if isinstance(bbox, (list, tuple)) else [bbox] for bbox in self.boxes],
            'top_emotion': self.get_top_emotions(),
            'top_score': [score for score in self.scores],
            'emotion_scores': [emotion for emotion in self.emotions],
        }
        return pd.DataFrame(data)