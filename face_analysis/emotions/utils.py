import base64
import os
import requests
from typing import Union, Optional
import logging

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from face_analysis.commons.logger import Logger
log = Logger()


class InvalidImage(Exception):
    pass


def loadBase64Img(uri):
    encoded_data = uri.split(",")[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def pil_to_bgr(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def load_image(img):
    """Modified from github.com/serengil/deepface. Returns bgr (opencv-style) numpy array."""
    is_exact_image = is_base64_img = is_url_img = False

    if type(img).__module__ == np.__name__:
        is_exact_image = True
    elif img is None:
        raise InvalidImage("Image not valid.")
    elif len(img) > 11 and img[0:11] == "data:image/":
        is_base64_img = True
    elif len(img) > 11 and img.startswith("http"):
        is_url_img = True

    if is_base64_img:
        img = loadBase64Img(img)
    elif is_url_img:
        img = pil_to_bgr(Image.open(requests.get(img, stream=True).raw))
    elif not is_exact_image:
        if not os.path.isfile(img):
            raise ValueError(f"Confirm that {img} exists")
        img = cv2.imread(img)

    if img is None or not hasattr(img, "shape"):
        raise InvalidImage("Image not valid.")

    return img


def setup_gpu(device: str = "gpu") -> str:
    device = device.lower()

    if device == "gpu" or device == "cuda":
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
                )
                
                log.info("Using GPU for processing")
                return "/GPU:0"
            except RuntimeError as e:
                log.warning(f"GPU Configuration Error: {e}")
        log.info("No GPU found, falling back to CPU")
        device = "cpu"

    # Force CPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        tf.config.set_visible_devices([], 'GPU')
    except RuntimeError as e:
        log.warning(f"Error disabling GPU: {e}")
    
    log.info("Using CPU for processing")
    return "/CPU:0"
