import logging
import json
import numpy as np
import torch
import cv2
from pathlib import Path



logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class FaceAnalysisPipeline:
    
    def __init__(self, config, output_dir= "output", device='cuda'):
        # Import the individual pipelines
        from .gazes import Pipeline as GazesPipeline
        from .eyes import Pipeline as EyesPipeline
        from .emotions import Pipeline as EmotionsPipeline

        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.gaze_pipeline = GazesPipeline(
            weights=config.L2CSNET_WEIGHTS,
            arch='ResNet50',
            device=self.device
        )
        
        self.eyes_pipeline = EyesPipeline(
            weights=config.EYE_STATE_MODEL_WEIGHTS,
            shape_predictor=config.SHAPE_PREDICTOR,
            device=device,
            include_detector=True
        )
        
        self.emotions_pipeline = EmotionsPipeline()
    
    def analyze_image(self, img_path, save_annotated_images=True):
        from .gazes import render as gazes_render
        from .eyes import render as eyes_render
        from .emotions import render as emotions_render
        from .emotions import load_image

        img_name = Path(img_path).stem
        img_path_str = str(img_path)
        
        image_bgr = cv2.imread(img_path_str)
        image_emotions = load_image(img_path_str)
        
        gaze_results = self.gaze_pipeline.step(image_bgr)
        eyes_results = self.eyes_pipeline.step(image_bgr)
        emotions_results = self.emotions_pipeline.step(image_emotions)
        
        annotated_gaze = gazes_render(image_bgr.copy(), gaze_results)
        annotated_eyes = eyes_render(image_bgr.copy(), eyes_results)
        annotated_emotions = emotions_render(image_emotions, emotions_results)
        
        if save_annotated_images:
            annotated_dir = self.output_dir / "annotated"
            annotated_dir.mkdir(exist_ok=True)
            gaze_img_path = annotated_dir / f"{img_name}_gazes.png"
            eyes_img_path = annotated_dir / f"{img_name}_eyes.png"
            emotions_img_path = annotated_dir / f"{img_name}_emotions.png"
            
            cv2.imwrite(str(gaze_img_path), annotated_gaze)
            cv2.imwrite(str(eyes_img_path), annotated_eyes)
            cv2.imwrite(str(emotions_img_path), annotated_emotions)
        
        all_results = self._unify_face_results(img_path_str, gaze_results, eyes_results, emotions_results)
        
        return all_results, {
            "gazes": annotated_gaze,
            "eyes": annotated_eyes,
            "emotions": annotated_emotions
        }
    
    def _unify_face_results(self, img_path, gaze_results, eyes_results, emotions_results):

        all_results = {img_path: {}}
        
        if len(gaze_results.bboxes) == 0:
            return all_results
            
        for face_idx, gaze_bbox in enumerate(gaze_results.bboxes):
            face_key = f"face_{face_idx}"
            all_results[img_path][face_key] = {
                "gazes": self._extract_gaze_data(gaze_results, face_idx),
                "eyes": self._find_matching_eyes_data(eyes_results, gaze_bbox),
                "emotions": self._find_matching_emotions_data(emotions_results, gaze_bbox)
            }
            
        return all_results
    
    def _extract_gaze_data(self, gaze_results, face_idx):

        if face_idx >= len(gaze_results.bboxes):
            return {}
            
        return {
            "bbox": gaze_results.bboxes[face_idx].tolist(),
            "landmarks": gaze_results.landmarks[face_idx].tolist(),
            "score": float(gaze_results.scores[face_idx]),
            "pitch": float(gaze_results.pitch[face_idx]),
            "yaw": float(gaze_results.yaw[face_idx])
        }
    
    def _find_matching_eyes_data(self, eyes_results, reference_bbox):
        if len(eyes_results.bboxes) == 0:
            return {}
            
        best_match_idx = self._find_best_bbox_match(reference_bbox, eyes_results.bboxes)
        
        if best_match_idx is None:
            return {}
            
        return {
            "bbox": eyes_results.bboxes[best_match_idx].tolist(),
            "landmarks": eyes_results.landmarks[best_match_idx].tolist(),
            "score": float(eyes_results.scores[best_match_idx]),
            "left_state": eyes_results.left_states[best_match_idx],
            "right_state": eyes_results.right_states[best_match_idx],
            "left_confidence": float(eyes_results.left_confidences[best_match_idx]),
            "right_confidence": float(eyes_results.right_confidences[best_match_idx]),
            "combined_state": eyes_results.get_combined_states()[best_match_idx]
        }
    
    def _find_matching_emotions_data(self, emotions_results, reference_bbox):

        if len(emotions_results.boxes) == 0:
            return {}
            
        emotion_bboxes = []
        for box in emotions_results.boxes:
            x, y, w, h = box
            emotion_bboxes.append([x, y, x+w, y+h])
            
        best_match_idx = self._find_best_bbox_match(reference_bbox, np.array(emotion_bboxes))
        
        if best_match_idx is None:
            return {}
            
        return {
            "bbox": emotions_results.boxes[best_match_idx],
            "emotions": emotions_results.emotions[best_match_idx],
            "top_emotion": emotions_results.get_top_emotions()[best_match_idx],
            "score": float(emotions_results.scores[best_match_idx])
        }
    
    def _find_best_bbox_match(self, reference_bbox, candidate_bboxes):
        if len(candidate_bboxes) == 0:
            return None
            
        best_iou = 0
        best_idx = None
        
        for idx, bbox in enumerate(candidate_bboxes):
            iou = self._calculate_iou(reference_bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        
        if best_iou > 0.5:
            return best_idx
        return None
    
    def _calculate_iou(self, box1, box2):

        if len(box1) == 4 and len(box2) == 4:

            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
                
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0.0
        else:
            return 0.0
    
    def save_results(self, all_results, output_path):
        with open((Path(output_path) / "face_analysis_results.json" ), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        log.info(f"Results saved to {output_path}")
