import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(BASE_DIR / "src"))

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_PATH = BASE_DIR / "output"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Model paths
SHAPE_PREDICTOR = MODEL_DIR / "shape_predictor_68_face_landmarks.dat"
EYE_STATE_MODEL_WEIGHTS = MODEL_DIR / "eye_state_classifier.h5"

# Parameters
GAZE_THRESHOLD = 8.0  # Angle threshold in degrees for considering looking at camera