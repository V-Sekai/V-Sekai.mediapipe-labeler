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
import os
import urllib.request
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from tqdm import tqdm
from cog import BasePredictor, Path, BaseModel
from PIL import Image, ImageDraw

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


############################################
# 1 Euro & Lowpass filter implementations
############################################
class LowPassFilter:
    """Implements a basic low-pass filter for smoothing values."""

    def __init__(self, alpha: float) -> None:
        self.__set_alpha(alpha)
        self.__y = self.__s = None

    def __set_alpha(self, alpha: float) -> None:
        if not 0 < alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1")
        self.__alpha = float(alpha)

    def __call__(self, value: float, alpha: float = None) -> float:
        if alpha:
            self.__set_alpha(alpha)
        self.__s = (
            value
            if self.__y is None
            else self.__alpha * value + (1.0 - self.__alpha) * self.__s
        )
        self.__y = value
        return self.__s

    @property
    def last_value(self) -> float:
        return self.__y


class OneEuroFilter:
    """Implements the 1€ filter for smoothing time-series data."""

    def __init__(
        self,
        freq: float = 30,
        mincutoff: float = 1.0,
        beta: float = 0.7,
        dcutoff: float = 1.0,
    ):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter(self.alpha(mincutoff))
        self.dx_filter = LowPassFilter(self.alpha(dcutoff))
        self.last_time = None

    def alpha(self, cutoff: float) -> float:
        return 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff * (1.0 / self.freq))))

    def __call__(self, x: float, timestamp: float = None) -> float:
        if self.last_time and timestamp:
            self.freq = 1.0 / (timestamp - self.last_time)
        self.last_time = timestamp or datetime.now().timestamp()

        dx = (
            (x - self.x_filter.last_value) * self.freq
            if self.x_filter.last_value is not None
            else 0.0
        )
        edx = self.dx_filter(dx)
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self.x_filter(x, alpha=self.alpha(cutoff))


############################################
# Constants and configuration
############################################
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


############################################
# Data models
############################################
class Output(BaseModel):
    """Output model for the predictor containing results and metadata."""

    coco_keypoints: str
    blendshapes: str
    debug_media: Path
    num_people: int
    media_type: str
    total_frames: Optional[int] = None


############################################
# Processing components
############################################
class PersonProcessor:
    """Handles person detection and crop processing."""

    @staticmethod
    def detect_people(
        image: np.ndarray,
        max_people: int,
        model_path: str = "thirdparty/ssd_mobilenet_v2.tflite",
    ) -> List[Tuple[int, int, int, int]]:
        """Detects people in an image using MobileNet SSD."""
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.4,
            category_allowlist=["person"],
            max_results=max_people,
        )
        detector = vision.ObjectDetector.create_from_options(options)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
        return [
            (
                detection.bounding_box.origin_x,
                detection.bounding_box.origin_y,
                detection.bounding_box.origin_x + detection.bounding_box.width,
                detection.bounding_box.origin_y + detection.bounding_box.height,
            )
            for detection in result.detections
        ]

    @staticmethod
    def process_crop(
        crop: np.ndarray,
        box: Tuple[int, int, int, int],
        original_size: Tuple[int, int],
        predictor: "Predictor",
    ) -> Optional[dict]:
        """Processes a crop of an image containing a person."""
        (start_x, start_y, end_x, end_y), (orig_height, orig_width) = box, original_size
        crop_height, crop_width = crop.shape[:2]

        try:
            pose_result = predictor.pose_processor.process(crop)
        except Exception as e:
            print(f"Processing error: {e}")
            return None

        def map_x(x: float) -> float:
            return (x / crop_width) * (end_x - start_x) + start_x

        def map_y(y: float) -> float:
            return (y / crop_height) * (end_y - start_y) + start_y

        mapped_pose = landmark_pb2.NormalizedLandmarkList()
        if pose_result.pose_landmarks:
            for lmk in pose_result.pose_landmarks.landmark:
                if lmk.visibility < 0.1:
                    mapped_pose.landmark.add(x=0, y=0, z=0, visibility=0)
                else:
                    mapped_pose.landmark.add(
                        x=map_x(lmk.x) / orig_width,
                        y=map_y(lmk.y) / orig_height,
                        z=lmk.z,
                        visibility=lmk.visibility,
                    )

        return {
            "coco": FullBodyProcessor.create_coco_output(mapped_pose, original_size)
        }


class FullBodyProcessor:
    """Handles COCO format conversion and processing."""

    @staticmethod
    def create_coco_output(
        pose: landmark_pb2.NormalizedLandmarkList, image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Creates COCO format output from pose landmarks."""
        height, width = image_size
        num_keypoints = len(COCO_KEYPOINT_NAMES)
        keypoints = [0.0] * (num_keypoints * 3)
        num_visible = 0

        if pose:
            for idx in range(17):
                if idx < len(pose.landmark):
                    lmk = pose.landmark[idx]
                    if lmk.visibility >= 0.1:
                        vis_flag = 2 if lmk.visibility > 0.5 else 1
                        keypoints[idx * 3] = lmk.x * width
                        keypoints[idx * 3 + 1] = lmk.y * height
                        keypoints[idx * 3 + 2] = vis_flag
                        num_visible += 1 if vis_flag == 2 else 0

        keypoint_objects = []
        for idx in range(num_keypoints):
            keypoint_objects.append(
                {
                    "id": idx,
                    "position": [keypoints[idx * 3], keypoints[idx * 3 + 1]],
                    "visibility": keypoints[idx * 3 + 2],
                    "parent": COCO_PARENT.get(idx, -1),
                }
            )

        return {
            "annotations": [
                {
                    "keypoint_objects": keypoint_objects,
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": num_visible,
                    "bbox": FullBodyProcessor.calculate_bbox(keypoints),
                    "area": width * height,
                }
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

    @staticmethod
    def calculate_bbox(keypoints: List[float]) -> List[float]:
        """Calculates bounding box from visible keypoints."""
        valid = [
            (keypoints[i], keypoints[i + 1])
            for i in range(0, len(keypoints), 3)
            if keypoints[i + 2] > 0
        ]
        if not valid:
            return [0.0] * 4
        x, y = zip(*valid)
        return [
            float(min(x)),
            float(min(y)),
            float(max(x) - min(x)),
            float(max(y) - min(y)),
        ]


############################################
# Main predictor
############################################
class Predictor(BasePredictor):
    """Main predictor class for pose estimation."""

    def setup(self) -> None:
        """Initializes models and resources."""
        os.makedirs("thirdparty", exist_ok=True)
        models = [
            (
                "ssd_mobilenet_v2.tflite",
                "https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float32/1/ssd_mobilenet_v2.tflite",
            ),
        ]
        for filename, url in models:
            path = f"thirdparty/{filename}"
            if not os.path.exists(path):
                urllib.request.urlretrieve(url, path)

        self.pose_processor = mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=2, min_detection_confidence=0.5
        )
        self.filters = [(OneEuroFilter(), OneEuroFilter()) for _ in COCO_KEYPOINT_NAMES]

    def apply_filters(self, results: List[Dict[str, Any]]) -> None:
        """Applies smoothing filters to keypoint coordinates."""
        ts = datetime.now().timestamp()
        for result in results:
            annotation = result["coco"]["annotations"][0]
            kps = annotation["keypoint_objects"]
            keypoints = annotation["keypoints"].copy()

            for kp in kps:
                idx = kp["id"]
                x, y = kp["position"]
                filt_x = self.filters[idx][0](x, ts)
                filt_y = self.filters[idx][1](y, ts)
                kp["position"] = [filt_x, filt_y]
                keypoints[idx * 3] = filt_x
                keypoints[idx * 3 + 1] = filt_y

            annotation["keypoints"] = keypoints

    def predict(
        self,
        media_path: Path,
        max_people: int = 20,
        frame_sample_rate: int = 1,
        max_processing_seconds: int = 0,
    ) -> Output:
        """Main prediction method."""
        if media_path.suffix.lower() in (".mp4", ".avi", ".mov"):
            return self.process_video(
                media_path, max_people, frame_sample_rate, max_processing_seconds
            )
        return self.process_image(media_path, max_people)

    def process_image(self, image_path: Path, max_people: int) -> Output:
        """Processes a single image."""
        img = np.array(Image.open(image_path).convert("RGB"))
        height, width = img.shape[:2]
        results = []

        for box in PersonProcessor.detect_people(img, max_people):
            crop = img[box[1] : box[3], box[0] : box[2]]
            if crop.size == 0:
                continue
            result = PersonProcessor.process_crop(crop, box, (height, width), self)
            if result:
                results.append(result)

        self.apply_filters(results)
        return Output(
            coco_keypoints=json.dumps(self.aggregate_coco(results, width, height)),
            blendshapes=json.dumps({"people": []}),
            debug_media=self.create_debug_image(img, results),
            num_people=len(results),
            media_type="image",
        )

    def process_video(
        self,
        video_path: Path,
        max_people: int,
        frame_sample_rate: int,
        max_processing_seconds: int,
    ) -> Output:
        """Processes a video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        debug_video = cv2.VideoWriter(
            "/tmp/debug.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS) / frame_sample_rate,
            (frame_width, frame_height),
        )

        frame_results = []
        max_people_count = 0
        progress = tqdm(total=total_frames, desc="Processing video")
        start_time = datetime.now()
        frame_count = 0

        while cap.isOpened():
            elapsed = (datetime.now() - start_time).total_seconds()
            if max_processing_seconds > 0 and elapsed > max_processing_seconds:
                print(f"Stopping after {max_processing_seconds} seconds")
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = frame_rgb.shape[:2]
                results = []

                for box in PersonProcessor.detect_people(frame_rgb, max_people):
                    crop = frame_rgb[box[1] : box[3], box[0] : box[2]]
                    result = PersonProcessor.process_crop(
                        crop, box, (height, width), self
                    )
                    if result:
                        results.append(result)

                self.apply_filters(results)
                max_people_count = max(max_people_count, len(results))
                frame_results.append(
                    {
                        "coco": self.aggregate_coco(results, width, height),
                        "blendshapes": [],
                    }
                )
                debug_frame = self.annotate_frame(frame_rgb, results)
                debug_video.write(
                    cv2.cvtColor(np.array(debug_frame), cv2.COLOR_RGB2BGR)
                )

            frame_count += 1
            progress.update(1)

        cap.release()
        debug_video.release()
        progress.close()

        return Output(
            coco_keypoints=json.dumps([f["coco"] for f in frame_results]),
            blendshapes=json.dumps({"frames": [{"people": []} for _ in frame_results]}),
            debug_media=Path("/tmp/debug.mp4"),
            num_people=max_people_count,
            media_type="video",
            total_frames=len(frame_results),
        )

    def draw_skeleton(
        self,
        draw: ImageDraw.Draw,
        keypoints: List[Dict[str, Any]],
        colors: Dict[str, Tuple[int, int, int]],
    ) -> None:
        """Draws skeleton connections and keypoints on an image."""
        kp_dict = {kp["id"]: kp for kp in keypoints}
        connections = []

        for kp in keypoints:
            parent_id = kp.get("parent", -1)
            if parent_id in kp_dict:
                connections.append((kp_dict[parent_id], kp))

        for parent, child in connections:
            if parent.get("visibility", 1) < 0.5 or child.get("visibility", 1) < 0.5:
                continue
            x1, y1 = parent["position"][0], parent["position"][1]
            x2, y2 = child["position"][0], child["position"][1]
            draw.line([(x1, y1), (x2, y2)], fill=colors["orange"], width=2)

        for kp in keypoints:
            if kp.get("visibility", 1) < 0.5:
                continue
            x, y = kp["position"]
            bbox = [(x - 4, y - 4), (x + 4, y + 4)]
            draw.ellipse(bbox, fill=colors["blue"], outline=None)

    def annotate_frame(
        self, frame: np.ndarray, results: List[Dict[str, Any]]
    ) -> Image.Image:
        """Annotates a frame with skeleton visualizations."""
        annotated = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated)
        colors = {"body": (0, 255, 0), "orange": (255, 165, 0), "blue": (0, 0, 255)}

        for result in results:
            if "coco" in result and result["coco"]["annotations"]:
                annotation = result["coco"]["annotations"][0]
                keypoint_objects = annotation.get("keypoint_objects", [])
                self.draw_skeleton(draw, keypoint_objects, colors)

        return annotated

    def create_debug_image(self, img: np.ndarray, results: list) -> Path:
        """Creates a debug image with visual annotations."""
        debug_img = self.annotate_frame(img, results)
        debug_img.save("/tmp/debug.jpg")
        return Path("/tmp/debug.jpg")

    def aggregate_coco(
        self, results: List[Dict[str, Any]], width: int, height: int
    ) -> Dict[str, Any]:
        """Aggregates COCO format results for output."""
        return {
            "info": {"description": "COCO Format Pose Estimation", "version": "1.0"},
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
            "annotations": [res["coco"]["annotations"][0] for res in results],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "keypoints": COCO_KEYPOINT_NAMES,
                    "skeleton": list(COCO_PARENT.items()),
                }
            ],
        }
