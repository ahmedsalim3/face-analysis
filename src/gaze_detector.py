import cv2
import numpy as np
import mediapipe as mp

from config import GAZE_THRESHOLD

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_LANDMARKS = [33, 133]  # Left eye corners
RIGHT_EYE_LANDMARKS = [362, 263]  # Right eye corners


def detect_gaze(image, gaze_threshold=GAZE_THRESHOLD):
    """Detect gaze direction for all faces in the image using MediaPipe Face Mesh.

    Args:
        image: Input image in BGR format
        gaze_threshold: Angle threshold in degrees for considering looking at camera

    Returns:
        List of dictionaries containing gaze information for each detected face
    """
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=10, min_detection_confidence=0.5
    )

    gaze_info = []
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate face bounding box from landmarks
            x_coords = [lm.x * image.shape[1] for lm in face_landmarks.landmark]
            y_coords = [lm.y * image.shape[0] for lm in face_landmarks.landmark]
            bbox = (
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords)),
            )

            # Extract eye landmarks
            left_eye = np.array(
                [
                    (
                        face_landmarks.landmark[idx].x * image.shape[1],
                        face_landmarks.landmark[idx].y * image.shape[0],
                    )
                    for idx in LEFT_EYE_LANDMARKS
                ]
            )

            right_eye = np.array(
                [
                    (
                        face_landmarks.landmark[idx].x * image.shape[1],
                        face_landmarks.landmark[idx].y * image.shape[0],
                    )
                    for idx in RIGHT_EYE_LANDMARKS
                ]
            )

            # Calculate gaze vectors and angles
            left_vector = left_eye[1] - left_eye[0]
            right_vector = right_eye[1] - right_eye[0]

            left_angle = np.degrees(np.arctan2(left_vector[1], left_vector[0]))
            right_angle = np.degrees(np.arctan2(right_vector[1], right_vector[0]))

            avg_gaze_angle = (abs(left_angle) + abs(right_angle)) / 2
            is_looking = avg_gaze_angle < gaze_threshold

            gaze_info.append(
                {
                    "bbox": bbox,
                    "is_looking": is_looking,
                    "gaze_angle": float(avg_gaze_angle),
                }
            )

    face_mesh.close()
    return gaze_info


def match_face_with_gaze(face, gaze_info, pixel_distance_threshold=50):
    """Match dlib face with mediapipe gaze info.

    Args:
        face: dlib face object
        gaze_info: List of gaze info dictionaries
        pixel_distance_threshold: Maximum distance for matching faces

    Returns:
        Matched gaze info or None
    """
    face_center = (
        (face.left() + face.right()) / 2,
        (face.top() + face.bottom()) / 2,
    )

    best_match = None
    min_distance = float("inf")

    for mp_face in gaze_info:
        mp_bbox = mp_face["bbox"]
        mp_center = ((mp_bbox[0] + mp_bbox[2]) / 2, (mp_bbox[1] + mp_bbox[3]) / 2)
        distance = np.sqrt(
            (face_center[0] - mp_center[0]) ** 2 + (face_center[1] - mp_center[1]) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            best_match = mp_face

    if (
        best_match and min_distance < pixel_distance_threshold
    ):  # pixel_distance_threshold =50
        return best_match
    return None
