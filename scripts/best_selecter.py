"""Select best image from folder based on face analysis scores"""

import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from image_analysis import analyze_image


def is_looking_at_camera(pitch, yaw, threshold=15):
    """Check if face is looking at camera"""
    return abs(pitch) < np.deg2rad(threshold) and abs(yaw) < np.deg2rad(threshold)


def score_image(results):
    """Score image based on face quality metrics"""
    if not results:
        return 0.0
    
    total_score = 0.0
    face_count = len(results)
    
    for face_data in results.values():
        # Eyes open score (40%)
        eyes = face_data.get("eyes", {})
        left_open = eyes.get("left_state") == "open"
        right_open = eyes.get("right_state") == "open"
        
        if left_open and right_open:
            eyes_score = 1.0
        elif left_open or right_open:
            eyes_score = 0.5
        else:
            eyes_score = 0.0
        
        # Looking at camera score (30%)
        gaze = face_data.get("gaze", {})
        pitch = gaze.get("pitch", 0)
        yaw = gaze.get("yaw", 0)
        looking_score = 1.0 if is_looking_at_camera(pitch, yaw) else 0.0
        
        # Emotion score (30%)
        emotions = face_data.get("emotions", {})
        top_emotion = emotions.get("top_emotion", "")
        smile_score = 1.0 if top_emotion == "happy" else 0.0
        
        face_score = (0.4 * eyes_score) + (0.3 * looking_score) + (0.3 * smile_score)
        total_score += face_score
    
    return total_score / face_count


def select_best_image(input_folder, output_folder, device='cpu', detector='retinaface'):
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    input_path = Path(input_folder)
    for ext in image_extensions:
        image_paths.extend(list(input_path.glob(f"*{ext}")))
        image_paths.extend(list(input_path.glob(f"*{ext.upper()}")))
    
    if not image_paths:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Analyzing {len(image_paths)} images...")
    
    scored_images = []
    
    for img_path in image_paths:
        print(f"Analyzing {img_path.name}")
        results = analyze_image(str(img_path), output_folder, device, detector, save_annotated=False)
        score = score_image(results)
        scored_images.append((img_path, score))
        print(f"  Score: {score:.3f}")
    
    # Sort by score (highest first)
    scored_images.sort(key=lambda x: x[1], reverse=True)
    
    # Copy best image
    best_image, best_score = scored_images[0]
    output_path = Path(output_folder) / f"BEST_{best_image.name}"
    shutil.copy2(best_image, output_path)
    
    # Save ranking
    ranking = [{"image": img.name, "score": score} for img, score in scored_images]
    with open(Path(output_folder) / "ranking.json", 'w') as f:
        json.dump(ranking, f, indent=2)
    
    print(f"\nBest image: {best_image.name} (score: {best_score:.3f})")
    print("\nFull ranking:")
    for i, (img, score) in enumerate(scored_images, 1):
        print(f"{i:2d}. {img.name}: {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='output')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--detector', default='retinaface')
    
    args = parser.parse_args()
    select_best_image(args.input, args.output, args.device, args.detector)
