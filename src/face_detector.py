from dataclasses import dataclass
import os
import dlib
import cv2
import numpy as np

from config import SHAPE_PREDICTOR


def create_face_detector():
    """
    Create and return face detector and predictor.
    """
    assert os.path.exists(
        SHAPE_PREDICTOR
    ), f"Shape predictor file does not exist at {SHAPE_PREDICTOR}, use the `scripts/shape_predictor.sh` file to download it."
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR))
    return detector, predictor


def extract_face_image(gray, face):
    """
    Extract and preprocess face region from the grayscale image.
    """
    face_img = gray[
        max(0, face.top()) : min(gray.shape[0], face.bottom()),
        max(0, face.left()) : min(gray.shape[1], face.right()),
    ]
    return cv2.resize(face_img, (100, 100))


######################
# NOT USED, BUT USEFUL
######################


@dataclass
class FaceDetectorConfig:
    scale_factor: float = 1.1
    min_face_size: int = 50
    min_neighbors: int = 5
    offsets: tuple = (10, 10)


def find_faces(
    img: np.ndarray,
    bgr=True,
    mtcnn=True,
    cascade_file=None,
    config: FaceDetectorConfig = FaceDetectorConfig(),
) -> list:
    if cascade_file is None:
        cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    if mtcnn:
        try:
            from facenet_pytorch import MTCNN
        except ImportError:
            raise Exception(
                "MTCNN not installed, install it with pip install facenet-pytorch and from facenet_pytorch import MTCNN"
            )
        __face_detector = "mtcnn"

        # use cuda GPU if available
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = torch.device("cuda")
            _mtcnn = MTCNN(keep_all=True, device=device)
        else:
            _mtcnn = MTCNN(keep_all=True)
    else:
        __face_detector = cv2.CascadeClassifier(cascade_file)

    if isinstance(__face_detector, cv2.CascadeClassifier):
        if bgr:
            gray_image_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:  # assume gray
            gray_image_array = img

        faces = __face_detector.detectMultiScale(
            gray_image_array,
            scaleFactor=config.scale_factor,
            minNeighbors=config.min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(config.min_face_size, config.min_face_size),
        )
    elif __face_detector == "mtcnn":
        boxes, probs = _mtcnn.detect(img)
        faces = []
        if isinstance(boxes, np.ndarray):
            for face in boxes:
                faces.append(
                    [
                        int(face[0]),
                        int(face[1]),
                        int(face[2]) - int(face[0]),
                        int(face[3]) - int(face[1]),
                    ]
                )

    return faces
