#!/usr/bin/env python3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import json
import math
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from cog import BasePredictor, Path, BaseModel
from PIL import Image, ImageDraw
import mediapipe as mp

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_PARENT = {
    0: -1,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 0,
    6: 0,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
    11: 5,
    12: 6,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
}


class Output(BaseModel):
    coco_keypoints: str
    blendshapes: str
    debug_media: Path
    num_people: int
    media_type: str
    total_frames: Optional[int] = None


class Predictor(BasePredictor):
    def setup(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=2, min_detection_confidence=0.5
        )
        self.filters = [(OneEuroFilter(), OneEuroFilter()) for _ in COCO_KEYPOINT_NAMES]

    def predict(self, media_path: Path, frame_sample: int = 1) -> Output:
        if media_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".mpv"):
            return self.process_video(media_path, frame_sample)
        return self.process_image(media_path)

    def process_image(self, image_path: Path) -> Output:
        img = np.array(Image.open(image_path).convert("RGB"))
        results = self.detect_poses(img)
        return Output(
            coco_keypoints=json.dumps(self.format_coco(results, *img.shape[:2])),
            blendshapes=json.dumps({"people": []}),
            debug_media=self.debug_image(img, results),
            num_people=len(results),
            media_type="image",
        )

    def process_video(self, video_path: Path, frame_sample: int) -> Output:
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        debug_video = cv2.VideoWriter(
            "/tmp/debug.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS) / frame_sample,
            (width, height),
        )

        frame_results = []
        for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret or frame_idx % frame_sample != 0:
                continue

            results = self.detect_poses(frame)
            debug_video.write(self.debug_image(frame, results))
            frame_results.append(results)

        cap.release()
        debug_video.release()

        return Output(
            coco_keypoints=json.dumps(
                [self.format_coco(r, width, height) for r in frame_results]
            ),
            blendshapes=json.dumps({"frames": []}),
            debug_media=Path("/tmp/debug.mp4"),
            num_people=max(len(r) for r in frame_results),
            media_type="video",
            total_frames=len(frame_results),
        )

    def detect_poses(self, image: np.ndarray) -> List[Dict]:
        results = []
        pose_data = self.pose.process(image)
        if pose_data.pose_landmarks:
            results.append(
                self.process_landmarks(pose_data.pose_landmarks, image.shape)
            )
        return results

    def process_landmarks(self, landmarks, img_shape):
        keypoints = []
        for idx, lmk in enumerate(landmarks.landmark):
            if lmk.visibility < 0.1:
                keypoints += [0.0] * 3
                continue

            x = lmk.x * img_shape[1]
            y = lmk.y * img_shape[0]
            vis = 2 if lmk.visibility > 0.5 else 1
            keypoints.extend([x, y, vis])

        return {
            "keypoints": keypoints,
            "bbox": self.calculate_bbox(keypoints),
            "annotations": [
                {
                    "id": idx,
                    "position": [keypoints[idx * 3], keypoints[idx * 3 + 1]],
                    "visibility": keypoints[idx * 3 + 2],
                    "parent": COCO_PARENT.get(idx, -1),
                }
                for idx in range(17)
            ],
        }

    def calculate_bbox(self, keypoints):
        valid = [
            (keypoints[i], keypoints[i + 1])
            for i in range(0, len(keypoints), 3)
            if keypoints[i + 2] > 0
        ]
        return (
            [
                min(x for x, y in valid),
                min(y for x, y in valid),
                max(x for x, y in valid) - min(x for x, y in valid),
                max(y for x, y in valid) - min(y for x, y in valid),
            ]
            if valid
            else [0] * 4
        )

    def format_coco(self, results, width, height):
        return {
            "info": {"description": "COCO Pose Results", "version": "1.0"},
            "licenses": [{"id": 1, "name": "CC-BY-4.0"}],
            "images": [
                {
                    "id": 0,
                    "width": width,
                    "height": height,
                    "file_name": "input",
                    "date_captured": datetime.now().isoformat(),
                }
            ],
            "annotations": [
                {
                    "id": i,
                    "image_id": 0,
                    "category_id": 1,
                    "iscrowd": 0,
                    "keypoints": r["keypoints"],
                    "num_keypoints": sum(1 for k in r["keypoints"][2::3] if k > 0),
                    "bbox": r["bbox"],
                    "area": width * height,
                }
                for i, r in enumerate(results)
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": COCO_KEYPOINT_NAMES,
                    "skeleton": list(COCO_PARENT.items()),
                }
            ],
        }

    def debug_image(self, frame, results):
        annotated = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated)
        for result in results:
            for kp in result["annotations"]:
                x, y = kp["position"]
                if kp["visibility"] < 0.5:
                    continue
                draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=(0, 0, 255))
                if kp["parent"] in result["annotations"]:
                    px, py = result["annotations"][kp["parent"]]["position"]
                    draw.line([(px, py), (x, y)], fill=(255, 165, 0), width=2)
        annotated.save("/tmp/debug.jpg")
        return np.array(annotated)


class OneEuroFilter:
    def __init__(self, freq=30, mincutoff=1.0, beta=0.7, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter(self.alpha(mincutoff))
        self.dx_filter = LowPassFilter(self.alpha(dcutoff))
        self.last_time = None

    def alpha(self, cutoff):
        return 1 / (1 + (1 / (2 * math.pi * cutoff * (1 / self.freq))))

    def __call__(self, x, timestamp=None):
        self.last_time = timestamp or datetime.now().timestamp()
        if self.x_filter.last_value is None:
            dx = 0
        else:
            dx = (x - self.x_filter.last_value) * self.freq

        edx = self.dx_filter(dx)
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self.x_filter(x, self.alpha(cutoff))


class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.last_value = None
        self.smoothed = None

    def __call__(self, value, alpha=None):
        if alpha:
            self.alpha = alpha
        if self.last_value is None:
            self.smoothed = value
        else:
            self.smoothed = self.alpha * value + (1 - self.alpha) * self.smoothed
        self.last_value = value
        return self.smoothed
