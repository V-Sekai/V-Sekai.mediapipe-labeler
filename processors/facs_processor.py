import math
from collections import defaultdict
from typing import List
from filters import OneEuroFilter

class FACSAnalyzer:
    def __init__(self):
        self.au_filters = defaultdict(lambda: OneEuroFilter(freq=30, mincutoff=0.8, beta=0.4))

    def calculate_au_vectors(self, landmarks, facs_config):
        vectors = {}
        for au, groups in facs_config["LANDMARK_MAPPING"].items():
            au_vectors = []
            for group_name, indices in groups.items():
                points = [landmarks[idx] for idx in indices if idx < len(landmarks)]
                if len(points) >= 2:
                    start = points[0]
                    end = points[1]
                    au_vectors.append({
                        "start": (start.x, start.y),
                        "end": (end.x, end.y),
                        "magnitude": math.hypot(end.x - start.x, end.y - start.y)
                    })
            vectors[au] = au_vectors
        return vectors

    def calculate_au_scores(self, blendshapes, facs_config):
        blendshape_dict = {bs.category_name: bs.score for bs in blendshapes}
        au_scores = {}
        for au, components in facs_config["AU_MAPPING"].items():
            scores = [blendshape_dict.get(name, 0.0) for name in components]
            au_scores[au] = sum(scores) / len(scores) if scores else 0.0
        return au_scores
