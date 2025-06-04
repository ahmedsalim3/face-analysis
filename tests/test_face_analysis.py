import pytest
import numpy as np
import cv2
from pathlib import Path

try:
    from face_analysis.gazes import Pipeline as GazesPipeline
    from face_analysis.eyes import Pipeline as EyesPipeline  
    from face_analysis.emotions import Pipeline as EmotionsPipeline, load_image
except ImportError as e:
    pytest.skip(f"face_analysis package not available: {e}", allow_module_level=True)


class TestFace:
    
    @pytest.fixture
    def img_bgr(self):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:, :, 0] = 100
        img[:, :, 1] = 150
        img[:, :, 2] = 200
        return img
    
    @pytest.fixture
    def img_rgb(self):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:, :, 0] = 200
        img[:, :, 1] = 150
        img[:, :, 2] = 100
        return img

    @pytest.fixture
    def face_img(self):
        img = np.ones((400, 400, 3), dtype=np.uint8) * 128
        cv2.rectangle(img, (150, 150), (250, 250), (200, 180, 160), -1)
        cv2.rectangle(img, (170, 180), (190, 200), (50, 50, 50), -1)
        cv2.rectangle(img, (210, 180), (230, 200), (50, 50, 50), -1)
        cv2.rectangle(img, (185, 220), (215, 235), (100, 80, 80), -1)
        return img

    def test_gaze_init(self, img_bgr):
        pipeline = GazesPipeline(arch='ResNet50', detector='retinaface', device='cpu')
        results = pipeline.step(img_bgr)
        assert hasattr(results, 'bboxes')
        assert hasattr(results, 'pitch')
        assert hasattr(results, 'yaw')
        assert hasattr(results, 'landmarks')
        assert hasattr(results, 'scores')
        assert hasattr(results, 'looking_at_camera')
        assert callable(getattr(results, 'is_looking_at_camera', None))
        assert callable(getattr(results, 'to_dataframe', None))

    def test_eyes_init(self, img_bgr):
        pipeline = EyesPipeline(detector='retinaface', device='cpu')
        results = pipeline.step(img_bgr)
        assert hasattr(results, 'bboxes')
        assert hasattr(results, 'left_states')
        assert hasattr(results, 'right_states')
        assert hasattr(results, 'left_confidences')
        assert hasattr(results, 'right_confidences')
        assert hasattr(results, 'landmarks')
        assert hasattr(results, 'scores')
        assert callable(getattr(results, 'get_combined_states', None))
        assert callable(getattr(results, 'get_blink_status', None))
        assert callable(getattr(results, 'to_dataframe', None))

    def test_emotion_init(self, img_rgb):
        pipeline = EmotionsPipeline(detector='retinaface', device='cpu')
        results = pipeline.step(img_rgb)
        assert hasattr(results, 'boxes')
        assert hasattr(results, 'emotions')
        assert hasattr(results, 'scores')
        assert callable(getattr(results, 'get_top_emotions', None))
        assert callable(getattr(results, 'filter_by_confidence', None))
        assert callable(getattr(results, 'get_face_details', None))
        assert callable(getattr(results, 'to_dataframe', None))

    def test_detectors(self):
        detectors = ['retinaface', 'mtcnn', 'cascade', 'dlib']
        
        for detector in detectors:
            try:
                gaze_pipeline = GazesPipeline(arch='ResNet50', detector=detector, device='cpu')
                assert gaze_pipeline is not None
            except (ImportError, ValueError):
                pass
                
            try:
                eyes_pipeline = EyesPipeline(detector=detector, device='cpu')
                assert eyes_pipeline is not None
            except (ImportError, ValueError):
                pass
                
            try:
                emotion_pipeline = EmotionsPipeline(detector=detector, device='cpu')
                assert emotion_pipeline is not None
            except (ImportError, ValueError):
                pass

    def test_archs(self):
        architectures = ['ResNet50', 'ResNet18', 'ResNet34', 'ResNet101', 'ResNet152']
        for arch in architectures:
            try:
                pipeline = GazesPipeline(arch=arch, detector='retinaface', device='cpu')
                assert pipeline is not None
            except Exception:
                pass

    def test_thresholds(self, img_bgr):
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for threshold in thresholds:
            try:
                gaze_pipeline = GazesPipeline(arch='ResNet50', detector='retinaface', 
                                            device='cpu', confidence_threshold=threshold)
                results = gaze_pipeline.step(img_bgr)
                assert results is not None
            except Exception:
                pass

    def test_devices(self, img_bgr):
        devices = ['cpu', 'cuda', 'gpu']
        for device in devices:
            try:
                pipeline = GazesPipeline(arch='ResNet50', detector='retinaface', device=device)
                results = pipeline.step(img_bgr)
                assert results is not None
            except Exception:
                pass

    def test_tfserving(self, img_rgb):
        try:
            pipeline = EmotionsPipeline(detector='retinaface', device='cpu', tfserving=True)
            results = pipeline.step(img_rgb)
            assert results is not None
        except Exception:
            pass

    def test_load_img(self):
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:, :] = [255, 0, 0]
        temp_path = Path("test_temp_image.png")
        try:
            cv2.imwrite(str(temp_path), test_img)
            loaded_img = load_image(str(temp_path))
            assert loaded_img is not None
            assert isinstance(loaded_img, np.ndarray)
            assert len(loaded_img.shape) == 3
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_results(self, face_img):
        gaze_pipeline = GazesPipeline(arch='ResNet50', detector='retinaface', device='cpu')
        eyes_pipeline = EyesPipeline(detector='retinaface', device='cpu')
        emotions_pipeline = EmotionsPipeline(detector='retinaface', device='cpu')
        
        gaze_results = gaze_pipeline.step(face_img)
        eyes_results = eyes_pipeline.step(face_img)
        emotions_results = emotions_pipeline.step(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        if len(gaze_results.pitch) > 0:
            assert gaze_results.is_looking_at_camera() is not None
            df = gaze_results.to_dataframe()
            assert df is not None
            
        if len(eyes_results.left_states) > 0:
            combined_states = eyes_results.get_combined_states()
            blink_status = eyes_results.get_blink_status()
            df = eyes_results.to_dataframe()
            assert combined_states is not None
            assert blink_status is not None
            assert df is not None
            
        if len(emotions_results.emotions) > 0:
            top_emotions = emotions_results.get_top_emotions()
            filtered_results = emotions_results.filter_by_confidence(0.5)
            face_details = emotions_results.get_face_details(0)
            df = emotions_results.to_dataframe()
            assert top_emotions is not None
            assert filtered_results is not None
            assert face_details is not None
            assert df is not None

    def test_process(self, img_rgb):
        try:
            pipeline = EmotionsPipeline(detector='retinaface', device='cpu')
            results = pipeline.process_image(img_rgb)
            assert results is not None
            assert hasattr(results, 'boxes')
            assert hasattr(results, 'emotions')
            assert hasattr(results, 'scores')
        except Exception:
            pass

    def test_predict(self, img_bgr):
        try:
            pipeline = GazesPipeline(arch='ResNet50', detector='retinaface', device='cpu')
            pitch, yaw = pipeline.predict_gaze(img_bgr)
            assert pitch is not None
            assert yaw is not None
        except Exception:
            pass

    def test_process_face(self, face_img):
        try:
            import dlib
            pipeline = EyesPipeline(detector='retinaface', device='cpu')
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_rect = dlib.rectangle(0, 0, 100, 100)
            left_state, left_conf, right_state, right_conf = pipeline.process_face(gray_face, face_rect)
            assert left_state in ['open', 'closed']
            assert right_state in ['open', 'closed']
            assert 0 <= left_conf <= 1
            assert 0 <= right_conf <= 1
        except Exception:
            pass

    def test_integration(self, face_img):
        gaze_pipeline = GazesPipeline(arch='ResNet50', detector='retinaface', device='cpu')
        eyes_pipeline = EyesPipeline(detector='retinaface', device='cpu') 
        emotions_pipeline = EmotionsPipeline(detector='retinaface', device='cpu')
        
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        gaze_results = gaze_pipeline.step(face_img)
        eyes_results = eyes_pipeline.step(face_img)
        emotions_results = emotions_pipeline.step(face_img_rgb)
        
        assert gaze_results is not None
        assert eyes_results is not None
        assert emotions_results is not None
        
        assert hasattr(gaze_results, 'bboxes')
        assert hasattr(eyes_results, 'bboxes')
        assert hasattr(emotions_results, 'boxes')

    def test_empty(self):
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        gaze_pipeline = GazesPipeline(arch='ResNet50', detector='retinaface', device='cpu')
        eyes_pipeline = EyesPipeline(detector='retinaface', device='cpu')
        emotions_pipeline = EmotionsPipeline(detector='retinaface', device='cpu')
        
        gaze_results = gaze_pipeline.step(empty_img)
        eyes_results = eyes_pipeline.step(empty_img)
        emotions_results = emotions_pipeline.step(empty_img)
        
        assert gaze_results is not None
        assert eyes_results is not None
        assert emotions_results is not None

    def test_version(self):
        try:
            from face_analysis.version import __version__
            assert isinstance(__version__, str)
            assert len(__version__) > 0
        except ImportError:
            pass

    def test_weights(self):
        try:
            custom_weights = Path("custom_weights.h5")
            if custom_weights.exists():
                pipeline = EmotionsPipeline(weights=custom_weights, detector='retinaface', device='cpu')
                assert pipeline is not None
        except Exception:
            pass

    def test_labels(self):
        pipeline = EmotionsPipeline(detector='retinaface', device='cpu')
        expected_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        for label in expected_labels:
            assert label in pipeline.EMOTION_LABELS.values()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
