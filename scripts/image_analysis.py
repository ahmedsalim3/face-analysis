"""Single image face analysis"""

import cv2
import json
import argparse
from pathlib import Path
from face_analysis.gazes import Pipeline as GazesPipeline, render as gazes_render
from face_analysis.eyes import Pipeline as EyesPipeline, render as eyes_render
from face_analysis.emotions import Pipeline as EmotionsPipeline, render as emotions_render, load_image


def analyze_image(img_path, output_dir, device='cpu', detector='retinaface', save_annotated=True, display=False):
    # Initialize pipelines
    gaze_pipeline = GazesPipeline(arch='ResNet50', detector=detector, device=device)
    eyes_pipeline = EyesPipeline(detector=detector if detector != "cascade" else "retinaface", device=device)
    emotions_pipeline = EmotionsPipeline(detector=detector, device=device)
    
    # Load images
    image_bgr = cv2.imread(img_path)
    image_emotions = load_image(img_path)
    
    # Run analysis
    gaze_results = gaze_pipeline.step(image_bgr)
    eyes_results = eyes_pipeline.step(image_bgr)
    emotions_results = emotions_pipeline.step(image_emotions)
    
    # Generate annotated images
    annotated = {
        "gazes": gazes_render(image_bgr.copy(), gaze_results),
        "eyes": eyes_render(image_bgr.copy(), eyes_results),
        "emotions": emotions_render(image_emotions, emotions_results)
    }
    
    # Save annotated images
    if save_annotated:
        img_name = Path(img_path).stem
        annotated_dir = Path(output_dir) / "annotated"
        annotated_dir.mkdir(exist_ok=True)
        
        for analysis_type, image in annotated.items():
            output_path = annotated_dir / f"{img_name}_{analysis_type}.png"
            cv2.imwrite(str(output_path), image)
    
    # Display results
    if display:
        for name, img in annotated.items():
            cv2.imshow(f"{name.capitalize()}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Prepare results
    results = {}
    if len(gaze_results.bboxes) > 0:
        for i, bbox in enumerate(gaze_results.bboxes):
            face_key = f"face_{i}"
            results[face_key] = {
                "bbox": bbox.tolist(),
                "gaze": {"pitch": float(gaze_results.pitch[i]), "yaw": float(gaze_results.yaw[i])},
                "eyes": get_eyes_data(eyes_results, i) if i < len(eyes_results.bboxes) else {},
                "emotions": get_emotions_data(emotions_results, i) if i < len(emotions_results.boxes) else {}
            }
    
    # Save results
    with open(Path(output_dir) / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def get_eyes_data(eyes_results, idx):
    if idx >= len(eyes_results.bboxes):
        return {}
    return {
        "left_state": eyes_results.left_states[idx],
        "right_state": eyes_results.right_states[idx],
        "left_confidence": float(eyes_results.left_confidences[idx]),
        "right_confidence": float(eyes_results.right_confidences[idx])
    }


def get_emotions_data(emotions_results, idx):
    if idx >= len(emotions_results.boxes):
        return {}
    return {
        "emotions": emotions_results.emotions[idx],
        "top_emotion": emotions_results.get_top_emotions()[idx],
        "score": float(emotions_results.scores[idx])
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='output')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--detector', default='retinaface')
    parser.add_argument('--save-annotated', action='store_true')
    parser.add_argument('--display', action='store_true')
    
    args = parser.parse_args()
    analyze_image(args.input, args.output, args.device, args.detector, args.save_annotated, args.display)
