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

import pathlib
from typing import Union
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
from face_detection import RetinaFace

try:
    from facenet_pytorch import MTCNN
    import torch
except ImportError:
    MTCNN = None

from .utils import prep_input_numpy, getArch
from .results import GazeResultContainer
from ..commons.get_weights import download_weights_if_necessary 

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gaze")

L2CSNET_WEIGHT_URL = "https://drive.google.com/uc?id=18S956r4jnHtSeT8z8t3z8AoJZjVnNqPJ"

class Pipeline:

    def __init__(
        self, 
        weights: pathlib.Path = None, 
        arch: str = "ResNet50",
        detector: str = "retinaface",
        device: str = "cpu",
        confidence_threshold: float = 0.5
        ):

        # Save input parameters
        self.weights = weights
        self.detector_type = detector.lower()
        self.confidence_threshold = confidence_threshold
        self.arch = arch
        
        if self.weights is None:
            self.weights = download_weights_if_necessary(
                file_name='L2CSNet_gaze360.pkl',
                source_url=L2CSNET_WEIGHT_URL,
                compress_type=None
            )

        # Set up device
        if device in ["gpu", "cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            log.info("No GPU detected, The default CPU will be used instead!")
            self.device = torch.device("cpu")
        
        # Initialize components
        self._initialize_face_detector()
        self._initialize_l2cs_model()
    
    def _initialize_face_detector(self):
        if self.detector_type == "mtcnn":
            if MTCNN is None:
                raise ImportError(
                    "MTCNN is not installed. Install it with pip install facenet-pytorch"
                )

            # Use CUDA if available for MTCNN
            if self.device.type == 'cuda':
                self.detector = MTCNN(keep_all=True, device=self.device)
            else:
                self.detector = MTCNN(keep_all=True)
        
        elif self.detector_type == "retinaface":
            # Create RetinaFace
            if self.device.type == 'cpu':
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(gpu_id=self.device.index)
        else:
            self.detector = None
            log.info("No valid face detector was specified. Using direct input mode.")
            
    def _initialize_l2cs_model(self):
        """Initialize the L2CS gaze estimation model."""
        self.model = getArch(self.arch, 90)
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = [idx for idx in range(90)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)
        
    def _find_faces(self, img: np.ndarray):
        """
        Detect faces in an image using the selected face detector,
        If no detector was given, return an empty faces list
        """
        faces = []
        
        if self.detector_type == "mtcnn":
            # MTCNN face detection
            boxes, probs = self.detector.detect(img)

            if boxes is not None:
                for i, face in enumerate(boxes):
                    # Apply conf threshold
                    if probs[i] < self.confidence_threshold:
                        continue

                    box = face.astype(np.int32)
                    # Create dummy landmark and use prob as score
                    landmark = np.zeros((5, 2))
                    score = float(probs[i])
                    faces.append([box, landmark, score])
            
        elif self.detector_type == "retinaface":
            # RetinaFace detection
            detections = self.detector(img)
            if detections:
                # Filter by threshold to get rid of low confidence faces
                faces = [face for face in detections if face[2] > self.confidence_threshold]
        
        return faces
        
    def step(self, frame: np.ndarray) -> GazeResultContainer:
        """
        Process a frame to detect faces and predict gaze direction
        """

        # Creating containers
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        # Find faces if we have a detector
        if self.detector is not None:
            faces = self._find_faces(frame)
            
            if faces:
                for box, landmark, score in faces:
                    # Extract safe min and max of x,y
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    
                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    # Save data
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                # Predict gaze
                pitch, yaw = self.predict_gaze(np.stack(face_imgs))

            else:
                # No faces detected
                pitch = np.empty((0,1))
                yaw = np.empty((0,1))
                bboxes = np.empty((0, 4))
                landmarks = np.empty((0, 5))
                scores = np.empty((0, 1))
        else:
            pitch, yaw = self.predict_gaze(frame)
            bboxes = np.empty((0, 4))
            landmarks = np.empty((0, 5))
            scores = np.empty((0, 1))

        # Save data
        if self.detector is not None and faces:
            results = GazeResultContainer(
                pitch=pitch,
                yaw=yaw,
                bboxes=np.stack(bboxes),
                landmarks=np.stack(landmarks),
                scores=np.stack(scores)
            )
        else:
            results = GazeResultContainer(
                pitch=pitch,
                yaw=yaw,
                bboxes=bboxes,
                landmarks=landmarks,
                scores=scores
            )

        return results

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):
        
        # Prepare input
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame
        else:
            raise RuntimeError("Invalid dtype for input")
    
        # Predict 
        gaze_pitch, gaze_yaw = self.model(img)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)
        
        # Get continuous predictions in degrees.
        pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        
        pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
        yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

        return pitch_predicted, yaw_predicted
