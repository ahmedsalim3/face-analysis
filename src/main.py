import logging
import os
import argparse
from pathlib import Path
import cv2

from face_detector import create_face_detector
from eye_state import load_eye_state_model, process_single_face
from gaze_detector import detect_gaze
from emotion_detector import detect_emotions
from utils import load_image, save_to_json, annotate_image, simplified_results
from config import RESULTS_PATH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler(RESULTS_PATH / "logs.log", mode='a')
    ],
)

def get_predictions(image_path, debug=False):
    """
    Analyze an image and return eye state predictions.

    Args:
        image_path: Path to the input image
        debug: Whether to show debug information

    Returns:
        Tuple containing:
        - Dictionary with prediction results for each face
        - The annotated image
    """

    img = load_image(image_path)
    model = load_eye_state_model()
    detector, predictor = create_face_detector()

    if debug:
        logging.info(f"Image shape: {img.shape}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    gaze_info = detect_gaze(img)
    
    emotions = detect_emotions(image_path)
    if debug:
        (x, y, w, h) = emotions["face_location"] if emotions["face_location"] else (0, 0, 0, 0)
        if (x, y, w, h) != (0, 0, 0, 0):
            annotated_img = img.copy()
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            top_emotion = emotions["emotions"]["top_emotion"]
            top_score = emotions["emotions"]["top_score"]
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{top_emotion}: {top_score:.2f}"
            cv2.putText(annotated_img, text, (x, y - 10), font, 0.9, (0, 255, 0), 2)
            cv2.imshow("Emotion", annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    face_results = {}
    annotated_img = img.copy()
    for i, face in enumerate(faces):
        face_data = process_single_face(face, gray, model, predictor, gaze_info, emotions, debug)
        face_results[f"face_{i}"] = face_data
        
        os.makedirs(RESULTS_PATH / "images", exist_ok=True)
        annotated_img = annotate_image(
            annotated_img, 
            face, 
            face_data,
            save_path=RESULTS_PATH / "images" / os.path.basename(image_path)
        )

    results = {"faces": face_results}
    
    if debug and len(faces) > 0:
        cv2.imshow("Annotated Image", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        

    return results, annotated_img

def process_image(image_path, save_results=True, debug=False):
    logging.info(f"Processing image: {image_path}")
    img_name = os.path.basename(image_path)    
    results, _ = get_predictions(image_path, debug=debug)
    
    full_results = {img_name: results}
    image_selection = simplified_results(img_name, results)
    
    if save_results:
        os.makedirs(RESULTS_PATH, exist_ok=True)
        save_to_json(full_results, RESULTS_PATH / "predictions.json")
        save_to_json(image_selection, RESULTS_PATH / "image_selection.json")
    
    if "error" not in results:
        logging.info(f"Image: {img_name}")
        for face_id, face_data in results["faces"].items():
            logging.info(f"\nFace {face_id}:")
            logging.info(f"Eyes open: {face_data['left_eye']['state'] == 'open' and face_data['right_eye']['state'] == 'open'}")
            logging.info(f"Looking at camera: {face_data['is_looking']}")
            if face_data.get('gaze_angle'):
                logging.info(f"Gaze angle: {face_data['gaze_angle']:.1f}°")

    return full_results, image_selection


def process_folder(folder_path, save_results=True, debug=False):
    """Process all images in a folder and save results to JSON."""
    full_results = {}
    all_simplified_results = {}
    if not os.path.isdir(folder_path):
        logging.error(f"The specified folder path {folder_path} does not exist.")
        return
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        image_results, image_selection = process_image(image_path, save_results=False, debug=debug)
        
        full_results.update(image_results)
        all_simplified_results.update(image_selection)
    
    if save_results:
        os.makedirs(RESULTS_PATH, exist_ok=True)
        save_to_json(full_results, RESULTS_PATH / "predictions.json")
        save_to_json(all_simplified_results, RESULTS_PATH / "image_selection.json")
    
    return full_results, all_simplified_results

def main():
    parser = argparse.ArgumentParser(description='Face Analysis Tool')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image or folder')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode with visualizations')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        logging.info(f"Processing all images in folder: {args.input}")
        process_folder(args.input, save_results=True, debug=args.debug)
    elif os.path.isfile(args.input):
        logging.info(f"Processing single image: {args.input}")
        process_image(args.input, save_results=True, debug=args.debug)
    else:
        logging.error(f"The specified path {args.input} does not exist or is invalid.")

if __name__ == "__main__":
    main()
