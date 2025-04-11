"""
Face Analysis Script
-------------------
Processes images or video with three face analysis pipelines:
1. Gaze Detection
2. Eye State Analysis
3. Emotion Recognition

Outputs visualization with subplots for each pipeline.
"""

from pathlib import Path
from types import SimpleNamespace
import cv2
import argparse
import logging
import time


config_dict = {
    'L2CSNET_WEIGHTS': 'models/L2CSNet_gaze360.pkl',
    'EYE_STATE_MODEL_WEIGHTS': 'models/eye_state_classifier.h5',
    'SHAPE_PREDICTOR': 'models/shape_predictor_68_face_landmarks.dat'
}

config = SimpleNamespace(**{key: Path(value) for key, value in config_dict.items()})
for f in vars(config).values():
    assert f.exists(), f"File does not exist: {f}"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger('demo')

try:
    from face_analysis.gazes import Pipeline as GazesPipeline
    from face_analysis.gazes import render as GazesRender
    from face_analysis.eyes import Pipeline as EyesPipeline
    from face_analysis.eyes import render as eyes_render
    from face_analysis.emotions import Pipeline as EmotionsPipeline
    from face_analysis.emotions import render as emotions_render
except ImportError:
    log.error("Failed to import face_analysis modules. Please ensure they are installed.")
    exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Face Analysis Script for Image or Video')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for processed image or video')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--detector', type=str, default='retinaface',
                        help='Face detector to use: "retinaface", "mtcnn", "dlib", or "cascade"')
    parser.add_argument('--display', action='store_true',
                        help='Display output while processing')
    return parser.parse_args()

def init_pipelines(args):
    log.info(f"Initializing pipelines on {args.device} using {args.detector} detector")
    try:
        # Gaze pipeline
        gaze_pipeline = GazesPipeline(
            weights=config.L2CSNET_WEIGHTS,
            arch='ResNet50',
            detector=args.detector,
            device=args.device,
        )
        log.info("Gaze pipeline initialized successfully")
        
        # Eye state pipeline
        eye_pipeline = EyesPipeline(
            weights=config.EYE_STATE_MODEL_WEIGHTS,
            shape_predictor=config.SHAPE_PREDICTOR,
            detector=args.detector if args.detector != "cascade" else "retinaface",
            device=args.device,
        )
        log.info("Eye state pipeline initialized successfully")
        
        # Emotion pipeline
        emotion_pipeline = EmotionsPipeline(
            detector=args.detector,
            device=args.device,
        )
        log.info("Emotion pipeline initialized successfully")
        
        return gaze_pipeline, eye_pipeline, emotion_pipeline
    
    except Exception as e:
        log.error(f"Error initializing pipelines: {str(e)}")
        exit(1)

def process_image(image_path, pipelines, output_path=None, display=False):
    """Process a single image with all pipelines."""
    gaze_pipeline, eye_pipeline, emotion_pipeline = pipelines
    
    img_in = cv2.imread(image_path)
    if img_in is None:
        log.error(f"Failed to read image: {image_path}")
        return
    
    log.info(f"Processing image: {image_path}")

    # Run all pipelines
    start = time.time()
    gaze_results = gaze_pipeline.step(img_in.copy())
    log.info(f"Gaze detection completed in {time.time() - start:.2f}s")

    start = time.time()
    eye_results = eye_pipeline.step(img_in.copy())
    log.info(f"Eye state analysis completed in {time.time() - start:.2f}s")

    start = time.time()
    emotion_results = emotion_pipeline.step(img_in.copy())
    log.info(f"Emotion recognition completed in {time.time() - start:.2f}s")

    # Render results
    gaze_img = GazesRender(img_in.copy(), gaze_results)
    eye_img = eyes_render(img_in.copy(), eye_results)
    emotion_img = emotions_render(img_in.copy(), emotion_results)

    if display:
        cv2.imshow("Original", img_in)
        cv2.imshow("Gaze", gaze_img)
        cv2.imshow("Eye State", eye_img)
        cv2.imshow("Emotion", emotion_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        img_name = Path(image_path).stem
        cv2.imwrite(output_path / f"{img_name}_original.jpg", img_in)
        cv2.imwrite(output_path / f"{img_name}_gaze.jpg", gaze_img)
        cv2.imwrite(output_path / f"{img_name}_eye_state.jpg", eye_img)
        cv2.imwrite(output_path / f"{img_name}_emotion.jpg", emotion_img)
        log.info(f"Results saved to {output_path}")

    return gaze_results, eye_results, emotion_results 


if __name__ == "__main__":
    args = parse_arguments()
    
    pipelines = init_pipelines(args)
    process_image(args.input, pipelines, args.output, args.display)
    
    log.info("Processing completed successfully")
