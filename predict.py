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
from tqdm import tqdm
import json
from PIL import ImageDraw, ImageFont
import math
import numpy as np
import os
import sys
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import Category, Landmark
from cog import BasePredictor, Input, Path, BaseModel
from PIL import Image
from typing import Any, Dict, List, Tuple, Optional, Union
from datetime import datetime
from pathlib import Path as PathLib


# 1€ Filter Implementation
class LowPassFilter:
    def __init__(self, alpha: float) -> None:
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha: float) -> None:
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError(f"alpha ({alpha}) should be in (0.0, 1.0]")
        self.__alpha = alpha

    def __call__(
        self, value: float, timestamp: float = None, alpha: float = None
    ) -> float:
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self) -> float:
        return self.__y

    def lastFilteredValue(self) -> float:
        return self.__s

    def reset(self) -> None:
        self.__y = None


class OneEuroFilter:
    def __init__(
        self,
        freq: float,
        mincutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0,
    ) -> None:
        if freq <= 0:
            raise ValueError("freq should be >0")
        if mincutoff <= 0:
            raise ValueError("mincutoff should be >0")
        if dcutoff <= 0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None

    def __alpha(self, cutoff: float) -> float:
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp: float = None) -> float:
        if self.__lasttime and timestamp and timestamp > self.__lasttime:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        prev_x = self.__x.lastFilteredValue()
        dx = 0.0 if prev_x is None else (x - prev_x) * self.__freq
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        cutoff = self.__mincutoff + self.__beta * math.fabs(edx)
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))


# Configuration
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
    "brow_inner_left",
    "brow_inner_right",
    "brow_outer_left",
    "brow_outer_right",
    "lid_upper_left",
    "lid_upper_right",
    "lid_lower_left",
    "lid_lower_right",
    "lip_upper",
    "lip_lower",
    "lip_corner_left",
    "lip_corner_right",
]

LEFT_HAND_VRM_MAPPING = {
    1: "leftThumbMetacarpal",
    2: "leftThumbProximal",
    3: "leftThumbDistal",
    5: "leftIndexProximal",
    6: "leftIndexIntermediate",
    7: "leftIndexDistal",
    9: "leftMiddleProximal",
    10: "leftMiddleIntermediate",
    11: "leftMiddleDistal",
    13: "leftRingProximal",
    14: "leftRingIntermediate",
    15: "leftRingDistal",
    17: "leftLittleProximal",
    18: "leftLittleIntermediate",
    19: "leftLittleDistal",
}

RIGHT_HAND_VRM_MAPPING = {
    1: "rightThumbMetacarpal",
    2: "rightThumbProximal",
    3: "rightThumbDistal",
    5: "rightIndexProximal",
    6: "rightIndexIntermediate",
    7: "rightIndexDistal",
    9: "rightMiddleProximal",
    10: "rightMiddleIntermediate",
    11: "rightMiddleDistal",
    13: "rightRingProximal",
    14: "rightRingIntermediate",
    15: "rightRingDistal",
    17: "rightLittleProximal",
    18: "rightLittleIntermediate",
    19: "rightLittleDistal",
}


class Output(BaseModel):
    coco_keypoints: str
    facs: str
    fullbodyfacs: str
    debug_media: Path
    hand_landmarks: Optional[str]
    num_people: int
    media_type: str
    total_frames: Optional[int] = None


class PersonProcessor:
    @staticmethod
    def detect_people(
        image_np: np.ndarray, max_people: int
    ) -> List[Tuple[int, int, int, int]]:
        base_options = python.BaseOptions(
            model_asset_path="thirdparty/ssd_mobilenet_v2.tflite",
            delegate=python.BaseOptions.Delegate.CPU,
        )
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.4,
            category_allowlist=["person"],
            max_results=max_people,
        )
        detector = vision.ObjectDetector.create_from_options(options)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        detection_result = detector.detect(mp_image)

        boxes = []
        h, w = image_np.shape[:2]

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            startX = int(bbox.origin_x)
            startY = int(bbox.origin_y)
            endX = int(bbox.origin_x + bbox.width)
            endY = int(bbox.origin_y + bbox.height)

            # Expand box by 20%
            width = endX - startX
            height = endY - startY
            startX = max(0, startX - int(0.2 * width))
            startY = max(0, startY - int(0.2 * height))
            endX = min(w, endX + int(0.2 * width))
            endY = min(h, endY + int(0.2 * height))

            boxes.append((startX, startY, endX, endY))

        return boxes

    @staticmethod
    def process_crop(
        crop: np.ndarray,
        box: Tuple[int, int, int, int],
        original_size: Tuple[int, int],
        predictor: Any,
    ):
        (startX, startY, endX, endY) = box
        orig_h, orig_w = original_size

        # Maintain aspect ratio
        h, w = crop.shape[:2]
        scale = 640 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (new_w, new_h))

        # Convert to RGB and process
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)

        try:
            face_result = predictor.face_processor.detect(mp_image)
            pose_result = predictor.pose_processor.process(resized)
            hand_result = predictor.hand_processor.detect(mp_image)
        except Exception as e:
            print(f"Processing error: {e}")
            return None

        # Coordinate mapping functions
        def map_x(x):
            return (x / new_w) * (endX - startX) + startX

        def map_y(y):
            return (y / new_h) * (endY - startY) + startY

        def map_z(z):
            return z * scale

        # Process pose landmarks with visibility filtering
        mapped_pose = None
        if pose_result.pose_landmarks:
            mapped_pose = landmark_pb2.NormalizedLandmarkList()
            for lmk in pose_result.pose_landmarks.landmark:
                if lmk.visibility < 0.1:  # Filter low-visibility keypoints
                    mapped_pose.landmark.append(
                        landmark_pb2.NormalizedLandmark(x=0, y=0, z=0, visibility=0)
                    )
                else:
                    mapped_pose.landmark.append(
                        landmark_pb2.NormalizedLandmark(
                            x=map_x(lmk.x * new_w) / orig_w,
                            y=map_y(lmk.y * new_h) / orig_h,
                            z=map_z(lmk.z),
                            visibility=lmk.visibility,
                        )
                    )

        # Process face landmarks with improved alignment
        mapped_face = None
        if face_result.face_landmarks:
            mapped_face = []
            for lmk in face_result.face_landmarks[0]:
                mapped_face.append(
                    Landmark(
                        x=map_x(lmk.x * new_w) / orig_w,
                        y=map_y(lmk.y * new_h) / orig_h,
                        z=map_z(lmk.z),
                    )
                )

        # Process hands
        left_hand, right_hand = None, None
        if hand_result.hand_landmarks:
            for idx, handedness in enumerate(hand_result.handedness):
                hand = []
                for lmk in hand_result.hand_landmarks[idx]:
                    hand.append(
                        Landmark(
                            x=map_x(lmk.x * new_w) / orig_w,
                            y=map_y(lmk.y * new_h) / orig_h,
                            z=map_z(lmk.z),
                        )
                    )
                if handedness[0].display_name == "Left":
                    left_hand = hand
                else:
                    right_hand = hand

        # Get blendshapes or empty list if none
        blendshapes = []
        if face_result.face_blendshapes:
            blendshapes = face_result.face_blendshapes[0]

        return FullBodyProcessor.process_results(
            mapped_pose,
            mapped_face,
            blendshapes,
            left_hand,
            right_hand,
            original_size,
        )


class FullBodyProcessor:
    @staticmethod
    def process_results(pose, face, blendshapes, left_hand, right_hand, image_size):
        return {
            "coco": FullBodyProcessor.create_coco_output(pose, face, image_size),
            "facs": FullBodyProcessor.create_facs_output(blendshapes),
            "fullbodyfacs": FullBodyProcessor.create_fullbodyfacs(
                pose, face, image_size
            ),
            "hands": FullBodyProcessor.process_hands(left_hand, right_hand),
        }

    @staticmethod
    def create_coco_output(pose, face, image_size):
        height, width = image_size
        keypoints = []
        num_visible = 0

        if pose:
            for idx in range(17):
                if idx < len(pose.landmark):
                    lmk = pose.landmark[idx]
                    vis = lmk.visibility
                    if vis < 0.1:  # Filter low-confidence keypoints
                        keypoints += [0.0, 0.0, 0]
                    else:
                        x = lmk.x * width
                        y = lmk.y * height
                        vis_flag = 2 if vis > 0.5 else 1
                        keypoints += [x, y, vis_flag]
                        num_visible += 1 if vis_flag == 2 else 0
                else:
                    keypoints += [0.0, 0.0, 0]

        # Updated facial indices based on MediaPipe's face landmark model
        facial_indices = [
            105,  # brow_inner_left
            334,  # brow_inner_right
            46,  # brow_outer_left
            276,  # brow_outer_right
            159,  # lid_upper_left
            386,  # lid_upper_right
            145,  # lid_lower_left
            374,  # lid_lower_right
            13,  # lip_upper
            14,  # lip_lower
            61,  # lip_corner_left
            291,  # lip_corner_right
        ]

        if face:
            for idx in facial_indices:
                if idx < len(face):
                    lmk = face[idx]
                    keypoints += [lmk.x * width, lmk.y * height, 2]
                    num_visible += 1
                else:
                    keypoints += [0.0, 0.0, 0]

        # Improved neck position calculation
        if pose and len(pose.landmark) > 12:
            nose_lmk = pose.landmark[0]
            left_shoulder = pose.landmark[11]
            right_shoulder = pose.landmark[12]

            # Only adjust if both shoulders are visible
            if left_shoulder.visibility >= 0.5 and right_shoulder.visibility >= 0.5:
                neck_x = (left_shoulder.x + right_shoulder.x) / 2 * width
                keypoints[0] = (nose_lmk.x * width + neck_x) / 2
            else:
                keypoints[0] = nose_lmk.x * width
            keypoints[1] = nose_lmk.y * height

        return {
            "annotations": [
                {
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
                    "skeleton": [
                        [16, 14],
                        [14, 12],
                        [17, 15],
                        [15, 13],
                        [12, 13],
                        [6, 12],
                        [7, 13],
                        [6, 7],
                        [6, 8],
                        [7, 9],
                        [8, 10],
                        [9, 11],
                        [2, 3],
                        [1, 2],
                        [1, 3],
                        [2, 4],
                        [3, 5],
                        [4, 6],
                        [5, 7],
                        [17, 18],
                        [18, 0],
                        [19, 17],
                        [20, 18],
                        [21, 17],
                        [22, 18],
                        [23, 21],
                        [24, 22],
                        [25, 27],
                        [26, 28],
                        [27, 25],
                        [28, 25],
                    ],
                }
            ],
        }

    @staticmethod
    def create_facs_output(blendshapes):
        au_scores = {}
        blendshape_dict = {}
        for bs in blendshapes:
            if hasattr(bs, "category_name") and hasattr(bs, "score"):
                blendshape_dict[bs.category_name] = bs.score

        FACS_AU_MAPPING = {
            "AU1": ["browInnerUp"],
            "AU2": ["browOuterUpLeft", "browOuterUpRight"],
            "AU4": ["browDownLeft", "browDownRight"],
            "AU5": ["eyeBlinkLeft", "eyeBlinkRight"],
            "AU6": ["eyeSquintLeft", "eyeSquintRight"],
            "AU9": ["noseSneerLeft", "noseSneerRight"],
            "AU12": ["mouthSmileLeft", "mouthSmileRight"],
            "AU25": ["jawOpen", "mouthStretch"],
        }

        for au, components in FACS_AU_MAPPING.items():
            scores = [blendshape_dict.get(name, 0.0) for name in components]
            au_scores[au] = sum(scores) / len(scores) if scores else 0.0

        return {
            "AUs": au_scores,
            "blendshapes": [
                {"name": bs.category_name, "score": float(bs.score)}
                for bs in blendshapes
                if hasattr(bs, "category_name") and hasattr(bs, "score")
            ],
        }

    @staticmethod
    def create_fullbodyfacs(pose, face, image_size):
        keypoints = []
        height, width = image_size
        left_shoulder = right_shoulder = None

        if pose:
            for idx, lmk in enumerate(pose.landmark):
                keypoints.append(
                    {
                        "id": idx,
                        "name": f"body_{idx}",
                        "position": [lmk.x * width, lmk.y * height, lmk.z * width],
                        "parent": FullBodyProcessor.get_parent(idx),
                        "visibility": lmk.visibility,
                    }
                )

                if idx == 11:
                    left_shoulder = lmk
                elif idx == 12:
                    right_shoulder = lmk

            # Improved neck calculation with visibility check
            if left_shoulder and right_shoulder:
                if left_shoulder.visibility >= 0.5 and right_shoulder.visibility >= 0.5:
                    neck_x = (left_shoulder.x + right_shoulder.x) / 2 * width
                    neck_y = (left_shoulder.y + right_shoulder.y) / 2 * height
                    neck_z = (left_shoulder.z + right_shoulder.z) / 2 * width
                    keypoints.append(
                        {
                            "id": 33,
                            "name": "neck",
                            "position": [neck_x, neck_y, neck_z],
                            "parent": 0,
                            "visibility": min(
                                left_shoulder.visibility, right_shoulder.visibility
                            ),
                        }
                    )

        facial_map = {
            105: 34,  # brow_inner_left
            334: 35,  # brow_inner_right
            46: 36,  # brow_outer_left
            276: 37,  # brow_outer_right
            159: 38,  # lid_upper_left
            386: 39,  # lid_upper_right
            145: 40,  # lid_lower_left
            374: 41,  # lid_lower_right
            13: 42,  # lip_upper
            14: 43,  # lip_lower
            61: 44,  # lip_corner_left
            291: 45,  # lip_corner_right
        }

        if face:
            for mp_idx, facs_id in facial_map.items():
                if mp_idx < len(face):
                    lmk = face[mp_idx]
                    keypoints.append(
                        {
                            "id": facs_id,
                            "name": COCO_KEYPOINT_NAMES[facs_id - 17],
                            "position": [lmk.x * width, lmk.y * height, lmk.z * width],
                            "parent": 0,
                            "visibility": 1.0,
                        }
                    )

        return {"keypoints": keypoints}

    @staticmethod
    def calculate_bbox(keypoints):
        valid = [
            (keypoints[i], keypoints[i + 1])
            for i in range(0, len(keypoints), 3)
            if keypoints[i + 2] > 0
        ]
        if not valid:
            return [0.0, 0.0, 0.0, 0.0]
        x = [p[0] for p in valid]
        y = [p[1] for p in valid]
        return [
            float(min(x)),
            float(min(y)),
            float(max(x) - min(x)),
            float(max(y) - min(y)),
        ]

    @staticmethod
    def process_hands(left_hand, right_hand):
        def process_single_hand(hand, is_left=True):
            if not hand:
                return []
            return [
                {
                    "index": idx,
                    "x": lmk.x,
                    "y": lmk.y,
                    "name": ("left_wrist" if is_left else "right_wrist")
                    if idx == 0
                    else (
                        LEFT_HAND_VRM_MAPPING if is_left else RIGHT_HAND_VRM_MAPPING
                    ).get(idx, None),
                }
                for idx, lmk in enumerate(hand)
            ]

        return {
            "left": process_single_hand(left_hand, True),
            "right": process_single_hand(right_hand, False),
        }

    @staticmethod
    def get_parent(idx):
        return {
            11: 33,
            12: 33,
            33: 0,
            13: 11,
            14: 12,
            15: 13,
            16: 14,
            23: 11,
            24: 12,
            25: 23,
            26: 24,
            27: 25,
            28: 26,
            34: 35,
            35: 0,
            36: 34,
            37: 35,
            38: 34,
            39: 35,
            40: 38,
            41: 39,
            42: 44,
            43: 45,
            44: 42,
            45: 42,
        }.get(idx, -1)


class Predictor(BasePredictor):
    def setup(self):
        os.makedirs("thirdparty", exist_ok=True)

        models = [
            (
                "ssd_mobilenet_v2.tflite",
                "https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float32/1/ssd_mobilenet_v2.tflite",
            ),
            (
                "face_landmarker.task",
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            ),
            (
                "hand_landmarker.task",
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            ),
        ]

        for filename, url in models:
            path = f"thirdparty/{filename}"
            if not os.path.exists(path):
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, path)

        self.face_processor = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="thirdparty/face_landmarker.task",
                    delegate=python.BaseOptions.Delegate.CPU,
                ),
                output_face_blendshapes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self.pose_processor = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            smooth_landmarks=True,
        )

        self.hand_processor = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="thirdparty/hand_landmarker.task",
                    delegate=python.BaseOptions.Delegate.CPU,
                ),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        # Initialize 1€ filters for each keypoint coordinate
        self.filters_x = []
        self.filters_y = []
        num_keypoints = len(COCO_KEYPOINT_NAMES)
        mincutoff = 1.0  # Lower values = more smoothing
        beta = 0.7  # Higher values = more responsive to movement
        dcutoff = 1.0
        initial_freq = 30

        for _ in range(num_keypoints):
            self.filters_x.append(OneEuroFilter(initial_freq, mincutoff, beta, dcutoff))
            self.filters_y.append(OneEuroFilter(initial_freq, mincutoff, beta, dcutoff))

    def predict(
        self,
        media_path: Path = Input(description="Input image or video file"),
        max_people: int = Input(
            description="Maximum number of people to detect (1-100)",
            ge=1,
            le=100,
            default=100,
        ),
        frame_sample_rate: int = Input(
            description="Process every nth frame for video input",
            ge=1,
            default=1,
        ),
    ) -> Output:
        media_type = "image"
        if str(media_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            media_type = "video"
            return self.process_video(media_path, max_people, frame_sample_rate)
        else:
            return self.process_image(media_path, max_people)

    def process_image(self, image_path: Path, max_people: int) -> Output:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        original_h, original_w = img_np.shape[:2]

        boxes = PersonProcessor.detect_people(img_np, max_people)
        all_results = []

        for person_id, box in enumerate(boxes):
            startX, startY, endX, endY = box
            crop = img_np[startY:endY, startX:endX]

            if crop.size == 0:
                continue

            person_result = PersonProcessor.process_crop(
                crop, box, (original_h, original_w), self
            )

            if person_result:
                person_result["person_id"] = person_id
                person_result["box"] = box
                all_results.append(person_result)

        return Output(
            coco_keypoints=json.dumps(
                self.aggregate_coco(all_results, original_w, original_h), indent=2
            ),
            facs=json.dumps({"people": [r["facs"] for r in all_results]}, indent=2),
            fullbodyfacs=json.dumps(
                {"people": [r["fullbodyfacs"] for r in all_results]}, indent=2
            ),
            debug_media=self.create_debug_image(img_np, all_results),
            hand_landmarks=json.dumps([r["hands"] for r in all_results])
            if any(r["hands"] for r in all_results)
            else None,
            num_people=len(all_results),
            media_type="image",
        )

    def process_video(
        self, video_path: Path, max_people: int, frame_sample_rate: int
    ) -> Output:
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup intermediate video writer (no audio)
        debug_video_path = "/tmp/debug_output_intermediate.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            debug_video_path, fourcc, fps / frame_sample_rate, (width, height)
        )

        # [Rest of video processing remains unchanged...]

        # Convert to Discord-compatible format with original audio
        final_video_path = "/tmp/debug_output_final.mp4"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # Overwrite output
                    "-i",
                    debug_video_path,  # Processed video
                    "-i",
                    str(video_path),  # Original video with audio
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-map",
                    "0:v:0",  # Video from processed
                    "-map",
                    "1:a:0",  # Audio from original
                    "-movflags",
                    "+faststart",
                    "-shortest",  # Match duration to shortest stream
                    final_video_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.remove(debug_video_path)  # Cleanup intermediate
        except Exception as e:
            print(f"Video conversion failed: {e}")
            final_video_path = debug_video_path  # Fallback

        return Output(
            # [Rest of output remains unchanged...]
            debug_media=Path(final_video_path),
            # [Other fields unchanged...]
        )

    def annotate_video_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
        annotated = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated)
        colors = {
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "red": (0, 0, 255),
            "orange": (255, 165, 0),
            "yellow": (255, 255, 0),
            "magenta": (255, 0, 255),
        }

        for result in results:
            startX, startY, endX, endY = result["box"]
            # Draw bounding box
            draw.rectangle(
                [(startX, startY), (endX, endY)], outline=colors["green"], width=2
            )

            # Draw skeleton and keypoints
            keypoints = result["fullbodyfacs"]["keypoints"]
            self.draw_skeleton(draw, keypoints, colors)

            # Draw person ID label
            label = f"Person {result['person_id']}"
            draw.text((startX, startY - 20), label, fill=colors["green"])

        return np.array(annotated)

    def create_debug_image(self, img_np: np.ndarray, all_results: list) -> Path:
        annotated = Image.fromarray(img_np)
        draw = ImageDraw.Draw(annotated)
        colors = {
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "red": (0, 0, 255),
            "orange": (255, 165, 0),
            "yellow": (255, 255, 0),
            "magenta": (255, 0, 255),
        }

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()

        for result in all_results:
            startX, startY, endX, endY = result["box"]
            draw.rectangle(
                [(startX, startY), (endX, endY)], outline=colors["green"], width=2
            )

            keypoints = result["fullbodyfacs"]["keypoints"]
            self.draw_skeleton(draw, keypoints, colors)

            label = f"Person {result['person_id']}"
            draw.text((startX, startY - 20), label, fill=colors["green"], font=font)

        debug_path = "/tmp/debug_output.jpg"
        annotated.save(debug_path)
        return Path(debug_path)

    def draw_skeleton(self, draw, keypoints, colors):
        kp_dict = {kp["id"]: kp for kp in keypoints}
        connections = []
        for kp in keypoints:
            parent_id = kp.get("parent", -1)
            if parent_id != -1 and parent_id in kp_dict:
                parent = kp_dict[parent_id]
                connections.append((parent, kp))

        for parent, child in connections:
            parent_vis = parent.get("visibility", 1.0)
            child_vis = child.get("visibility", 1.0)
            if parent_vis < 0.5 or child_vis < 0.5:
                continue
            x1, y1 = parent["position"][0], parent["position"][1]
            x2, y2 = child["position"][0], child["position"][1]
            draw.line([(x1, y1), (x2, y2)], fill=colors["orange"], width=2)

        for kp in keypoints:
            x, y = kp["position"][0], kp["position"][1]
            bbox = [(x - 4, y - 4), (x + 4, y + 4)]
            color = colors["blue"] if kp["id"] < 33 else colors["red"]
            draw.ellipse(bbox, fill=color, outline=None)

    def aggregate_coco(self, results, width, height):
        annotations = []
        for idx, res in enumerate(results):
            ann = res["coco"]["annotations"][0].copy()
            ann["id"] = idx
            annotations.append(ann)

        return {
            "info": {
                "description": "Multi-person COCO 1.1 Extended",
                "version": "1.1",
                "year": 2023,
                "contributor": "MediaPipe Crowd Processor",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [{"id": 1, "name": "CC-BY-4.0"}],
            "images": [
                {
                    "id": 0,
                    "width": width,
                    "height": height,
                    "file_name": "input.jpg",
                    "license": 1,
                    "date_captured": datetime.now().isoformat(),
                }
            ],
            "annotations": annotations,
            "categories": FullBodyProcessor.create_coco_output(None, None, (0, 0))[
                "categories"
            ],
        }


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    media_path = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    result = predictor.predict(media_path)
    print(json.dumps(json.loads(result.json()), indent=2))
