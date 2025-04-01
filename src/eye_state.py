import cv2
import numpy as np

from model import EyeStateClassifierNet
from utils import get_left_eye_attributes, get_right_eye_attributes
from face_detector import extract_face_image
from gaze_detector import match_face_with_gaze
from config import EYE_STATE_MODEL_WEIGHTS


def load_eye_state_model(weights_path=EYE_STATE_MODEL_WEIGHTS):
    model = EyeStateClassifierNet(compile=True).model
    model.load_weights(weights_path)
    return model


def prepare_eye_data(eye_img, keypoints, distances, angles):

    img = eye_img.reshape(-1, 24, 24, 1).astype(np.float32) / 255
    kp = np.expand_dims(keypoints, 1).astype(np.float32) / 24
    d = np.expand_dims(distances, 1).astype(np.float32) / 24
    a = np.expand_dims(angles, 1).astype(np.float32) / np.pi

    kp = kp.reshape(-1, 1, 11, 2)
    d = d.reshape(-1, 1, 11, 1)
    a = a.reshape(-1, 1, 11, 1)

    return img, kp, d, a


def predict_eye_state(model, eye_data):

    img, kp, d, a = eye_data
    prediction = model.predict([img, kp, d, a])[0]
    arg_max = np.argmax(prediction)

    state = "open" if arg_max == 1 else "closed"
    confidence = float(prediction[arg_max])

    return state, confidence


def process_single_face(
    face, gray_img, model, predictor, gaze_info, emotion_data, debug=False
):
    """Process a single face and predict eye states.

    Args:
        face: dlib face object
        gray_img: Grayscale image
        model: Eye state classifier model
        predictor: dlib shape predictor
        gaze_info: List of gaze info dictionaries
        emotion_data: Emotion detection results
        debug: Whether to show debug visualizations

    Returns:
        Dictionary with face results
    """

    # Extract and preprocess face
    face_img = extract_face_image(gray_img, face)

    # Get eye attributes
    l_i, lkp, ld, la = get_left_eye_attributes(face_img, predictor, (24, 24, 1))
    r_i, rkp, rd, ra = get_right_eye_attributes(face_img, predictor, (24, 24, 1))

    if debug:
        cv2.imshow(f"Left eye", l_i)
        cv2.imshow(f"Right eye", r_i)

    # Prepare eye data
    left_eye_data = prepare_eye_data(l_i, lkp, ld, la)
    right_eye_data = prepare_eye_data(r_i, rkp, rd, ra)

    # Predict eye states
    left_state, left_confidence = predict_eye_state(model, left_eye_data)
    right_state, right_confidence = predict_eye_state(model, right_eye_data)

    # Match with gaze info
    matched_gaze = match_face_with_gaze(face, gaze_info)

    # Create result dictionary
    face_results = {
        "left_eye": {
            "state": left_state,
            "confidence": left_confidence,
        },
        "right_eye": {
            "state": right_state,
            "confidence": right_confidence,
        },
        "face_bbox": {
            "left": face.left(),
            "top": face.top(),
            "right": face.right(),
            "bottom": face.bottom(),
        },
        "is_looking": False,  # Default value
    }

    # Add gaze info if available
    if matched_gaze:
        face_results["is_looking"] = bool(matched_gaze["is_looking"])
        face_results["gaze_angle"] = matched_gaze["gaze_angle"]

    # Add emotion data
    face_results["emotion_data"] = emotion_data

    return face_results
