from PIL import ImageDraw, ImageFont
import numpy as np
from typing import Dict, List

class AnnotationUtils:
    COLORS = {
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "red": (0, 0, 255),
        "orange": (255, 165, 0),
        "yellow": (255, 255, 0),
        "magenta": (255, 0, 255)
    }

    @classmethod
    def draw_skeleton(cls, draw, keypoints: List[Dict], colors: Dict):
        kp_dict = {kp["id"]: kp for kp in keypoints}
        connections = []
        for kp in keypoints:
            parent_id = kp.get("parent", -1)
            if parent_id in kp_dict:
                connections.append((kp_dict[parent_id], kp))

        for parent, child in connections:
            if parent["visibility"] < 0.5 or child["visibility"] < 0.5:
                continue
            x1, y1 = parent["position"][0], parent["position"][1]
            x2, y2 = child["position"][0], child["position"][1]
            draw.line([(x1, y1), (x2, y2)], fill=colors["orange"], width=2)

        for kp in keypoints:
            x, y = kp["position"][0], kp["position"][1]
            bbox = [(x-4, y-4), (x+4, y+4)]
            color = colors["blue"] if kp["id"] < 33 else colors["red"]
            draw.ellipse(bbox, fill=color, outline=None)

    @classmethod
    def create_annotated_frame(cls, frame: np.ndarray, results: list) -> np.ndarray:
        annotated = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated)
        for result in results:
            startX, startY, endX, endY = result["box"]
            draw.rectangle([(startX, startY), (endX, endY)], outline=cls.COLORS["green"], width=2)
            cls.draw_skeleton(draw, result["fullbodyfacs"]["keypoints"], cls.COLORS)
            draw.text((startX, startY-20), f"Person {result['person_id']}", fill=cls.COLORS["green"])
        return np.array(annotated)
