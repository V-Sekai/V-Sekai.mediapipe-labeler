import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from typing import List, Tuple, Optional, Any

class PoseProcessor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            smooth_landmarks=True
        )

    def process(self, image: np.ndarray) -> Any:
        return self.pose.process(image)
