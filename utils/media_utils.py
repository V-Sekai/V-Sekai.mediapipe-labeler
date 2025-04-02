import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict

class MediaUtils:
    @staticmethod
    def detect_people(image_np: np.ndarray, model_path: str, max_people: int) -> List[Tuple[int, int, int, int]]:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.4,
            category_allowlist=["person"],
            max_results=max_people
        )
        detector = vision.ObjectDetector.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        detection_result = detector.detect(mp_image)
        return MediaUtils._process_detections(detection_result, image_np.shape)

    @staticmethod
    def _process_detections(detection_result, image_shape):
        boxes = []
        h, w = image_shape[:2]
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            startX = int(bbox.origin_x)
            startY = int(bbox.origin_y)
            endX = int(bbox.origin_x + bbox.width)
            endY = int(bbox.origin_y + bbox.height)
            width = endX - startX
            height = endY - startY
            startX = max(0, startX - int(0.2 * width))
            startY = max(0, startY - int(0.2 * height))
            endX = min(w, endX + int(0.2 * width))
            endY = min(h, endY + int(0.2 * height))
            boxes.append((startX, startY, endX, endY))
        return boxes

    @staticmethod
    def prepare_crop(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        scale = 640 / max(w, h)
        return cv2.resize(crop, (int(w * scale), int(h * scale)))
