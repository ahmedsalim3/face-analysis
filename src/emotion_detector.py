from fer import FER
import numpy as np


def detect_emotions(image_path):
    """
    Detect emotions in an image using FER:
    pip install fer

    ref: https://github.com/JustinShenk/fer/blob/master/src/fer/fer.py
    """
    detector = FER(mtcnn=True)
    emotion_scores = detector.detect_emotions(image_path)

    if emotion_scores:
        emotion_data = emotion_scores[0]["emotions"]
        face_axis = emotion_scores[0]["box"]
        if isinstance(face_axis, np.ndarray):
            face_axis = face_axis.tolist()

        top_emotion, top_score = detector.top_emotion(image_path)

        result = {
            "emotions": {
                "top_emotion": top_emotion,
                "top_score": float(top_score) if top_score else None,
                "scores": {
                    "angry": float(emotion_data.get("angry", 0)),
                    "disgust": float(emotion_data.get("disgust", 0)),
                    "fear": float(emotion_data.get("fear", 0)),
                    "happy": float(emotion_data.get("happy", 0)),
                    "sad": float(emotion_data.get("sad", 0)),
                    "surprise": float(emotion_data.get("surprise", 0)),
                    "neutral": float(emotion_data.get("neutral", 0)),
                },
            },
            "face_location": face_axis,
        }
    else:
        result = {
            "emotions": {
                "top_emotion": None,
                "top_score": None,
                "scores": {
                    "angry": 0.0,
                    "disgust": 0.0,
                    "fear": 0.0,
                    "happy": 0.0,
                    "sad": 0.0,
                    "surprise": 0.0,
                    "neutral": 0.0,
                },
            },
            "face_location": None,
        }

    return result
