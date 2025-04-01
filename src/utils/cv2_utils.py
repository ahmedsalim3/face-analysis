import logging
import os
import cv2


def load_image(image_path):
    """
    Load an image from path.
    """
    if not os.path.exists(image_path):
        logging.error(f"Image not found: {image_path}")
        return None

    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to read image: {image_path}")
            return None
        return img
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None


def annotate_image(image, face, face_data, save_path=None):
    """Annotate image with face detection results.

    Args:
        image: Input image
        face: dlib face object
        face_data: Face detection results
        save_path: Path to save the annotated image

    Returns:
        Annotated image
    """
    img_copy = image.copy()

    # face bbox
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    cv2.rectangle(img_copy, (left, top), (right, bottom), (0, 255, 0), 2)

    # text for eye states
    left_eye_state = face_data["left_eye"]["state"]
    right_eye_state = face_data["right_eye"]["state"]
    is_looking = face_data.get("is_looking", False)

    text_color = (
        (0, 255, 0)
        if (left_eye_state == "open" and right_eye_state == "open")
        else (0, 0, 255)
    )
    cv2.putText(
        img_copy,
        f"L: {left_eye_state}",
        (left, top - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
    )
    cv2.putText(
        img_copy,
        f"R: {right_eye_state}",
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
    )

    # text for gaze estimation
    gaze_color = (0, 255, 255) if is_looking else (0, 0, 255)
    gaze_text = "Looking" if is_looking else "Not looking"
    if "gaze_angle" in face_data:
        gaze_text  # += f" ({face_data['gaze_angle']:.1f}°)"
    cv2.putText(
        img_copy,
        gaze_text,
        (left, bottom + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        gaze_color,
        2,
    )

    # emotion info
    if (
        "emotion_data" in face_data
        and face_data["emotion_data"]["emotions"]["top_emotion"]
    ):
        emotion = face_data["emotion_data"]["emotions"]["top_emotion"]
        score = face_data["emotion_data"]["emotions"]["top_score"]
        emotion_text = f"{emotion}: {score:.2f}"
        cv2.putText(
            img_copy,
            emotion_text,
            (left, bottom + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(str(save_path), img_copy)

    return img_copy
