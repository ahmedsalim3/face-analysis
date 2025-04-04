import cv2
import numpy as np

from .results import EmotionResultContainer

color_map = {
    "angry": (0, 0, 255),  # Red
    "disgust": (0, 140, 255),  # Orange
    "fear": (0, 255, 255),  # Yellow
    "happy": (0, 255, 0),  # Green
    "sad": (255, 0, 0),  # Blue
    "surprise": (255, 0, 255),  # Purple
    "neutral": (255, 255, 255),  # White
}


def render(
    frame: np.ndarray, results: EmotionResultContainer, show_details: bool = False
) -> np.ndarray:
    
    image_out = frame.copy()
    num_faces = len(results)

    if num_faces == 0:
        return image_out

    top_emotions = results.get_top_emotions()

    for i in range(num_faces):
        box = results.boxes[i]
        emotions = results.emotions[i]
        top_emotion = top_emotions[i]

        color = color_map.get(top_emotion, (255, 255, 255))
        x, y, w, h = box
        cv2.rectangle(image_out, (x, y), (x + w, y + h), color, 2)

        confidence = emotions[top_emotion]
        label = f"{top_emotion}: {confidence:.2f}"
        cv2.putText(
            image_out,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return image_out


def draw_emotion(
    frame: np.ndarray, bbox, emotions_dict: dict, show_details: bool = False
):
    output = frame.copy()

    top_emotion = max(emotions_dict, key=emotions_dict.get)
    color = color_map.get(top_emotion, (255, 255, 255))

    x, y, w, h = bbox
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    confidence = emotions_dict[top_emotion]
    label = f"{top_emotion}: {confidence:.2f}"
    cv2.putText(
        output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
    )

    if show_details:
        y_offset = y + h + 15
        for emotion, score in sorted(
            emotions_dict.items(), key=lambda x: x[1], reverse=True
        ):
            if score < 0.01:
                continue

            cv2.putText(
                output,
                f"{emotion}: {score:.2f}",
                (x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            y_offset += 15

    return output
