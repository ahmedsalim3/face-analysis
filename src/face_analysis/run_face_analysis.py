import logging
import argparse
from pathlib import Path
from types import SimpleNamespace

from face_analysis.face_analysis_pipline import FaceAnalysisPipeline

config_dict = {
    'L2CSNET_WEIGHTS': 'models/L2CSNet_gaze360.pkl',
    'EYE_STATE_MODEL_WEIGHTS': 'models/eye_state_classifier.h5',
    'SHAPE_PREDICTOR': 'models/shape_predictor_68_face_landmarks.dat'
}

config = SimpleNamespace(**{key: Path(value) for key, value in config_dict.items()})
for f in vars(config).values():
    assert f.exists(), f"File does not exist: {f}"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def analyze_images(image_paths, config, output_dir="output", device="cuda", save_annotated_images=True):
    
    pipeline = FaceAnalysisPipeline(config, output_dir, device=device)
    
    all_results = {}
    
    for img_path in image_paths:
        log.info(f"Analyzing {img_path}...")
        results, _ = pipeline.analyze_image(img_path, save_annotated_images)
        all_results.update(results)
    
    pipeline.save_results(all_results, output_dir)
    
    log.info(f"Analysis complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run combined face analysis pipeline')
    parser.add_argument('--images', nargs='+', default=["input/test_1.jpg"], 
                        help='List of image paths to analyze')
    parser.add_argument('--output', default='output/face_analysis_results.json',
                        help='Output JSON file path')
    parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--save-annotated-images', action='store_true',
                        help="Save all the annotated images")
    
    args = parser.parse_args()    
    image_paths = [str(Path(img_path).resolve()) for img_path in args.images]
    
    # Run analysis
    analyze_images(image_paths, config, args.output, args.device, args.save_annotated_images)

if __name__ == "__main__":
    main()
