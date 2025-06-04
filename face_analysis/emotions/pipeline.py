# MIT License

# Copyright (c) 2025 Ahmed Salim

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# =============================================================
# This code is based on the FER model and has been extended with
# RetinaFace face detection.
#
# The original FER repo can be found at:
# https://github.com/JustinShenk/fer/blob/master/src/fer/fer.py

import pathlib
from typing import Union, Tuple, List, Optional

import cv2
import numpy as np
from tensorflow.keras.models import load_model


try:
    from facenet_pytorch import MTCNN
    import torch
except ImportError:
    MTCNN = None


import tensorflow as tf
tf.get_logger().setLevel('ERROR')

try:
    from face_detection import RetinaFace
except ImportError:
    from face_analysis import RetinaFace

from face_analysis.commons.logger import Logger
from face_analysis.emotions.utils import load_image, setup_gpu
from face_analysis.emotions.results import EmotionResultContainer

log = Logger()

class Pipeline:

    EMOTION_LABELS = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral",
    }

    def __init__(
        self,
        weights: Optional[pathlib.Path] = None,
        device: str = "cpu",
        detector: str = "retinaface",
        confidence_threshold: float = 0.5,
        scale_factor: float = 1.1,
        min_face_size: int = 50,
        min_neighbors: int = 5,
        offsets: tuple = (10, 10),
        tfserving: bool = False,
        server_url: str = "http://localhost:8501/v1/models/emotion_model:predict",
    ):

        self.device = setup_gpu(device)
        self.detector_type = detector.lower()
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        self.min_face_size = min_face_size
        self.min_neighbors = min_neighbors
        self.offsets = offsets
        self.tfserving = tfserving
        self.server_url = server_url

        self._initialize_face_detector()
        self._initialize_emotion_model(weights)

    def _initialize_face_detector(self):
        """Initialize the selected face detector."""

        if self.detector_type == "mtcnn":
            if MTCNN is None:
                raise ImportError(
                    "MTCNN is not installed. Install it with pip install facenet-pytorch"
                )

            # Use CUDA if available for MTCNN
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device = torch.device("cuda")
                self.detector = MTCNN(keep_all=True, device=device)
            else:
                self.detector = MTCNN(keep_all=True)

        elif self.detector_type == "cascade":
            cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_file)

        elif self.detector_type == "retinaface":
            if "CPU" in self.device:
                self.detector = RetinaFace()
            else:
                gpu_id = 0
                self.detector = RetinaFace(gpu_id=gpu_id)

    def _initialize_emotion_model(self, weights: Optional[pathlib.Path] = None):
        if self.tfserving:
            self.emotion_target_size = (64, 64)
        else:
            if weights is None:
                # Use the default model from FER package
                emotion_model = pathlib.Path(__file__).parent / 'data' / 'emotion_model.hdf5'
            else:
                emotion_model = str(weights)

            # log.info(f"Loading emotion model from: {emotion_model}")
            self.emotion_classifier = load_model(emotion_model, compile=False)
            self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

    def find_faces(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image using the selected face detector
        """
        faces = []
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img

        if self.detector_type == "cascade":
            # OpenCV Cascade face detection
            detections = self.detector.detectMultiScale(
                gray_img,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(self.min_face_size, self.min_face_size),
            )

            # Convert to list of (x, y, w, h)
            if len(detections) > 0:
                faces = [(x, y, w, h) for (x, y, w, h) in detections]

        elif self.detector_type == "mtcnn":
            # MTCNN face detection
            boxes, probs = self.detector.detect(img)

            if boxes is not None:
                for i, face in enumerate(boxes):
                    # Apply conf threshold
                    if probs[i] < self.confidence_threshold:
                        continue

                    # Convert to (x,y,w,h)
                    x1, y1, x2, y2 = map(int, face)
                    w, h = x2 - x1, y2 - y1
                    faces.append((x1, y1, w, h))

        else:
            # RetinaFace
            detections = self.detector(img)

            if detections is not None:
                for box, landmarks, score in detections:
                    if score < self.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    faces.append((x1, y1, w, h))

        return faces

    def _preprocess_input(self, x, v2=True):
        x = x.astype("float32")
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    def _apply_offsets(self, face_coordinates):
        x, y, width, height = face_coordinates
        x_off, y_off = self.offsets

        x1 = max(0, x - x_off)
        y1 = max(0, y - y_off)
        x2 = x + width + x_off
        y2 = y + height + y_off

        return x1, x2, y1, y2

    def _to_square(self, bbox):
        x, y, w, h = bbox
        if h > w:
            diff = h - w
            x -= diff // 2
            w += diff
        elif w > h:
            diff = w - h
            y -= diff // 2
            h += diff

        return (x, y, w, h)

    def _classify_emotions(self, gray_faces: np.ndarray) -> np.ndarray:
        if self.tfserving:
            import requests

            gray_faces = np.expand_dims(gray_faces, -1)
            instances = gray_faces.tolist()
            response = requests.post(self.server_url, json={"instances": instances})
            response.raise_for_status()

            emotion_predictions = response.json()["predictions"]
            return np.array(emotion_predictions)
        else:
            return self.emotion_classifier.predict(gray_faces, verbose=0)

    def process_image(self, img: Union[str, np.ndarray]) -> EmotionResultContainer:
        """
        Process an image to detect faces and recognize emotions

        Returns:
            EmotionResultContainer with emotion detection results
        """

        image = load_image(img)

        boxes = []
        emotions_list = []
        scores = []

        face_rectangles = self.find_faces(image)

        if face_rectangles:

            gray_img = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) > 2
                else image
            )
            gray_faces = []
            for face_coordinates in face_rectangles:

                face_coordinates = self._to_square(face_coordinates)
                boxes.append(face_coordinates)

                x1, x2, y1, y2 = self._apply_offsets(face_coordinates)

                try:
                    gray_face = gray_img[y1:y2, x1:x2]
                    gray_face = cv2.resize(gray_face, self.emotion_target_size)
                    gray_face = self._preprocess_input(gray_face, True)
                    gray_faces.append(gray_face)
                except Exception as e:
                    log.warning(f"Face preprocessing failed: {e}")
                    continue

            if gray_faces:

                if len(gray_faces) == 1:
                    gray_faces = np.expand_dims(gray_faces[0], axis=0)
                else:
                    gray_faces = np.array(gray_faces)

                if len(gray_faces.shape) == 3:
                    gray_faces = np.expand_dims(gray_faces, -1)

                emotion_predictions = self._classify_emotions(gray_faces)

                for face_idx, preds in enumerate(emotion_predictions):
                    emotions = {
                        self.EMOTION_LABELS[i]: float(score)
                        for i, score in enumerate(preds)
                    }
                    top_emotion = max(emotions, key=emotions.get)
                    top_score = emotions[top_emotion]

                    emotions_list.append(emotions)
                    scores.append(top_score)

        results = EmotionResultContainer(
            boxes=boxes,
            emotions=emotions_list,
            scores=scores,
        )

        return results

    def step(self, frame: np.ndarray) -> EmotionResultContainer:
        return self.process_image(frame)
