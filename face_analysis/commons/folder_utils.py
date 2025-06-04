import os
from face_analysis.commons.logger import Logger

logger = Logger()


def initialize_folder() -> None:
    """
    Initialize the folder for storing model weights.

    Raises:
        OSError: if the folder cannot be created.
    """
    home = _home_path()
    home_path = os.path.join(home, ".face_analysis")
    weights_path = os.path.join(home_path, "weights")

    if not os.path.exists(home_path):
        os.makedirs(home_path, exist_ok=True)
        logger.info(f"Directory {home_path} has been created")

    if not os.path.exists(weights_path):
        os.makedirs(weights_path, exist_ok=True)
        logger.info(f"Directory {weights_path} has been created")


def _home_path() -> str:
    """
    Get the home directory for storing model weights

    Returns:
        str: the home directory.
    """
    return str(os.getenv("FACE_ANALYSIS_HOME", default=os.path.expanduser("~")))


def _weight_path(target_weight) -> str:
    """
    Get the path to the weights directory.

    Returns:
        str: the path to the weights directory.
    """
    home = _home_path()
    return os.path.join(home, ".face_analysis", "weights", target_weight)
