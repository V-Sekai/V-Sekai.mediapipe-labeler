from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceProcessor:
    def __init__(self, model_path: str):
        self.detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                output_face_blendshapes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

    def detect(self, image):
        return self.detector.detect(image)
