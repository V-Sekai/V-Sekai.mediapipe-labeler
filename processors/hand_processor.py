from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandProcessor:
    def __init__(self, model_path: str):
        self.detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

    def detect(self, image):
        return self.detector.detect(image)
