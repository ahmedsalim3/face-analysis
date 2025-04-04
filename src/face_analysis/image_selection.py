import logging
import json
import shutil
import numpy as np
from pathlib import Path
import argparse

from face_analysis.face_analysis_pipline import FaceAnalysisPipeline
import face_analysis.config as config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("face_analysis.image_selection")

def is_looking_at_camera(pitch, yaw, threshold=15):
    if abs(pitch) < np.deg2rad(threshold) and abs(yaw) < np.deg2rad(threshold):
        return True
    return False

def simplify_face_data(face_data):
    gazes = face_data.get("gazes", {})
    bbox = gazes.get("bbox", [])
    pitch = gazes.get("pitch", 0)
    yaw = gazes.get("yaw", 0)
    looking_at_camera = 1 if is_looking_at_camera(pitch, yaw) else 0
    
    # Extract eye data
    eyes = face_data.get("eyes", {})
    left_state = eyes.get("left_state", "unknown")
    left_confidence = eyes.get("left_confidence", 0)
    right_state = eyes.get("right_state", "unknown")
    right_confidence = eyes.get("right_confidence", 0)
    
    # Extract emotion data
    emotions = face_data.get("emotions", {})
    top_emotion = emotions.get("top_emotion", "unknown")
    emotion_score = emotions.get("score", 0)
    
    # Log the extracted face data
    log.info(f"Face data - Looking at camera: {looking_at_camera} " +
             f"(pitch: {np.rad2deg(pitch):.2f}°, yaw: {np.rad2deg(yaw):.2f}°)")
    log.info(f"Face data - Eyes: Left {left_state} (conf: {left_confidence:.4f}), " +
             f"Right {right_state} (conf: {right_confidence:.4f})")
    log.info(f"Face data - Emotion: {top_emotion} (score: {emotion_score:.4f})")
    
    # Compile simplified data
    simplified = {
        "bbox": bbox,
        "left_eye": {
            "state": left_state,
            "confidence": left_confidence
        },
        "right_eye": {
            "state": right_state,
            "confidence": right_confidence
        },
        "looking_at_camera": looking_at_camera,
        "emotion": {
            "type": top_emotion,
            "confidence": emotion_score
        }
    }
    
    return simplified

def calculate_image_score(simplified_faces):
    if not simplified_faces:
        return 0.0
    
    total_score = 0.0
    
    for i, face in enumerate(simplified_faces):
        # Eyes open score (40%)
        eyes_score = 0.0
        left_eye_open = face["left_eye"]["state"] == "open"
        right_eye_open = face["right_eye"]["state"] == "open"
        
        if left_eye_open and right_eye_open:
            eyes_score = 1.0
        elif left_eye_open or right_eye_open:
            eyes_score = 0.5
        
        # Looking at camera score (30%)
        looking_score = face["looking_at_camera"]
        
        # Smile score (30%)
        smile_score = 1.0 if face["emotion"]["type"] == "happy" else 0.0
        
        # Weighted score for this face
        face_score = (0.4 * eyes_score) + (0.3 * looking_score) + (0.3 * smile_score)
        
        log.info(f"Face {i+1} scores - Eyes: {eyes_score:.2f} (weight: 40%), " +
                 f"Looking: {looking_score:.2f} (weight: 30%), " +
                 f"Smile: {smile_score:.2f} (weight: 30%)")
        log.info(f"Face {i+1} weighted score: {face_score:.4f}")
        
        total_score += face_score
    
    # Average score across all faces
    avg_score = total_score / len(simplified_faces)
    log.info(f"Average score across {len(simplified_faces)} faces: {avg_score:.4f}")
    return avg_score

def process_images(input_folder, output_folder, device="cuda", save_annotated_images=True):
    log.info(f"Starting image processing from {input_folder} to {output_folder}")
    log.info(f"Using device: {device}, Save annotated images: {save_annotated_images}")

    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    pipeline = FaceAnalysisPipeline(config, output_dir=output_folder, device=device)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(Path(input_folder).glob(f"*{ext}")))
    
    if not image_paths:
        log.info(f"No images found in {input_folder}")
        return
    
    log.info(f"Found {len(image_paths)} images to analyze")
    
    best_score = -1
    best_image_path = None
    all_results = {}
    scored_images = []
    
    for img_path in image_paths:
        img_name = Path(img_path).stem
        img_path_str = str(img_path)
        log.info(f"\n===== Analyzing {img_name} =====")
        
        results, _ = pipeline.analyze_image(img_path_str, save_annotated_images)
        
        if not results or img_path_str not in results or not results[img_path_str]:
            log.info(f"No faces detected in {img_name}")
            continue
        
        faces_data = results[img_path_str]
        log.info(f"Found {len(faces_data)} faces in {img_name}")
        
        simplified_faces = []
        for face_key, face_data in faces_data.items():
            log.info(f"\nProcessing {face_key} in {img_name}")
            simplified = simplify_face_data(face_data)
            simplified_faces.append(simplified)
        
        log.info(f"\nCalculating overall score for {img_name}")
        score = calculate_image_score(simplified_faces)
        log.info(f"Final score for {img_path.name}: {score:.4f}")
        
        all_results[img_path_str] = {
            "faces": simplified_faces,
            "score": score
        }
        
        scored_images.append((img_path, score))
        
        if score > best_score:
            best_score = score
            best_image_path = img_path
            log.info(f"New best image: {img_path.name} with score {score:.4f}")
    
    scored_images.sort(key=lambda x: x[1], reverse=True)
    log.info("\n===== Image Ranking =====")
    for idx, (path, score) in enumerate(scored_images):
        log.info(f"{idx+1}. {path.name}: {score:.4f}")
    
    if best_image_path:
        best_image_output = output_dir / best_image_path.name
        shutil.copy2(best_image_path, best_image_output)
        all_results["best_image"] = best_image_path.name
        log.info(f"\nBest image is {best_image_path.name} with score {best_score:.4f}")
        log.info(f"Copied best image to {best_image_output}")
    else:
        log.info("\nNo suitable images found")
    
    json_path = output_dir / "image_selection.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log.info(f"Results saved to {json_path}")

def main():
    parser = argparse.ArgumentParser(description='Select best images based on face analysis')
    parser.add_argument('--input', required=True, help='Input folder containing images')
    parser.add_argument('--output', default='image_selection', help='Output folder for results')
    parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'], 
                        help='Device to run inference on')
    parser.add_argument('--save-annotated-images', action='store_true',
                        help="Save all the annotated images")
    
    args = parser.parse_args()
    
    log.info(f"Starting face analysis with args: {args}")
    process_images(args.input, args.output, args.device, args.save_annotated_images)

if __name__ == "__main__":
    main()