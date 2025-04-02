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

import tempfile
import cv2
from tqdm import tqdm
import json
from PIL import ImageDraw, ImageFont
import math
import numpy as np
import os
import sys
from moviepy import VideoFileClip
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


MEDIAPIPE_KEYPOINT_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
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
    mediapipe_keypoints: Path
    blendshapes: Path
    fullbody_data: Path
    debug_media: Path
    hand_landmarks: Optional[Path]
    num_people: int
    media_type: str
    total_frames: Optional[int] = None


class PersonProcessor:
    @staticmethod
    def detect_people(
        image_np: np.ndarray, max_people: int
    ) -> List[Tuple[int, int, int, int]]:
        base_options = python.BaseOptions(
            model_asset_path="thirdparty/ssd_mobilenet_v2.tflite"
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
        h, w = crop.shape[:2]
        scale = 640 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (new_w, new_h))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)

        try:
            face_result = predictor.face_processor.detect(mp_image)
            pose_result = predictor.pose_processor.process(resized)
            hand_result = predictor.hand_processor.detect(mp_image)
        except Exception as e:
            print(f"Processing error: {e}")
            return None

        def map_x(x):
            return (x / new_w) * (endX - startX) + startX

        def map_y(y):
            return (y / new_h) * (endY - startY) + startY

        def map_z(z):
            return z * scale

        mapped_pose = None
        if pose_result.pose_landmarks:
            mapped_pose = landmark_pb2.NormalizedLandmarkList()
            for lmk in pose_result.pose_landmarks.landmark:
                if lmk.visibility < 0.1:
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

        mapped_face = []
        if face_result.face_landmarks:
            mapped_face = [
                Landmark(
                    x=map_x(lmk.x * new_w) / orig_w,
                    y=map_y(lmk.y * new_h) / orig_h,
                    z=map_z(lmk.z),
                )
                for lmk in face_result.face_landmarks[0]
            ]

        left_hand, right_hand = None, None
        if hand_result.hand_landmarks:
            for idx, handedness in enumerate(hand_result.handedness):
                hand = [
                    Landmark(
                        x=map_x(lmk.x * new_w) / orig_w,
                        y=map_y(lmk.y * new_h) / orig_h,
                        z=map_z(lmk.z),
                    )
                    for lmk in hand_result.hand_landmarks[idx]
                ]
                if handedness[0].display_name == "Left":
                    left_hand = hand
                else:
                    right_hand = hand

        blendshapes = (
            face_result.face_blendshapes[0] if face_result.face_blendshapes else []
        )
        return FullBodyProcessor.process_results(
            mapped_pose, mapped_face, blendshapes, left_hand, right_hand, original_size
        )


class FullBodyProcessor:
    SKELETON_CONNECTIONS = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 7],
        [0, 4],
        [4, 5],
        [5, 6],
        [6, 8],
        [9, 10],
        [11, 12],
        [12, 14],
        [14, 16],
        [11, 13],
        [13, 15],
        [15, 17],
        [12, 24],
        [24, 26],
        [26, 28],
        [11, 23],
        [23, 25],
        [25, 27],
        [28, 30],
        [30, 32],
        [27, 29],
        [29, 31],
    ]

    @staticmethod
    def process_results(pose, face, blendshapes, left_hand, right_hand, image_size):
        return {
            "mediapipe": FullBodyProcessor.create_mediapipe_output(
                pose, face, image_size
            ),
            "blendshapes": FullBodyProcessor.create_blendshapes_output(blendshapes),
            "fullbody": FullBodyProcessor.create_fullbody_output(
                pose, face, image_size
            ),
            "hands": FullBodyProcessor.process_hands(left_hand, right_hand),
        }

    @staticmethod
    def create_mediapipe_output(pose, face, image_size):
        height, width = image_size
        keypoints = []
        num_visible = 0

        if pose:
            for lmk in pose.landmark[:33]:
                vis = lmk.visibility
                if vis < 0.1:
                    keypoints += [0.0, 0.0, 0]
                else:
                    x = lmk.x * width
                    y = lmk.y * height
                    vis_flag = 2 if vis > 0.5 else 1
                    keypoints += [x, y, vis_flag]
                    num_visible += 1 if vis_flag == 2 else 0
        else:
            keypoints += [0.0, 0.0, 0] * 33

        facial_indices = [105, 334, 46, 276, 159, 386, 145, 374, 13, 14, 61, 291]
        if face:
            for idx in facial_indices:
                if idx < len(face):
                    lmk = face[idx]
                    keypoints += [lmk.x * width, lmk.y * height, 2]
                    num_visible += 1
                else:
                    keypoints += [0.0, 0.0, 0]
        else:
            keypoints += [0.0, 0.0, 0] * 12

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
                    "keypoints": MEDIAPIPE_KEYPOINT_NAMES,
                    "skeleton": FullBodyProcessor.SKELETON_CONNECTIONS,
                }
            ],
        }

    @staticmethod
    def create_blendshapes_output(blendshapes):
        return [
            {"name": bs.category_name, "score": float(bs.score)}
            for bs in blendshapes
            if hasattr(bs, "category_name") and hasattr(bs, "score")
        ]

    @staticmethod
    def create_fullbody_output(pose, face, image_size):
        keypoints = []
        height, width = image_size

        if pose:
            for idx, lmk in enumerate(pose.landmark[:33]):
                keypoints.append(
                    {
                        "id": idx,
                        "name": MEDIAPIPE_KEYPOINT_NAMES[idx],
                        "position": [lmk.x * width, lmk.y * height, lmk.z * width],
                        "visibility": lmk.visibility,
                    }
                )

        facial_map = {
            105: 33,
            334: 34,
            46: 35,
            276: 36,
            159: 37,
            386: 38,
            145: 39,
            374: 40,
            13: 41,
            14: 42,
            61: 43,
            291: 44,
        }
        if face:
            for mp_idx, body_id in facial_map.items():
                if mp_idx < len(face):
                    lmk = face[mp_idx]
                    keypoints.append(
                        {
                            "id": body_id,
                            "name": MEDIAPIPE_KEYPOINT_NAMES[body_id],
                            "position": [lmk.x * width, lmk.y * height, lmk.z * width],
                            "visibility": 0.0,
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
                    "z": lmk.z,  # Fixed: Added z-coordinate
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

        # Initialize face processor and get blendshape names
        self.face_processor = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="thirdparty/face_landmarker.task"
                ),
                output_face_blendshapes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
        )
        dummy_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.zeros((100, 100, 3), dtype=np.uint8),
        )
        face_result = self.face_processor.detect(dummy_image)
        self.blendshape_names = (
            [bs.category_name for bs in face_result.face_blendshapes[0]]
            if face_result.face_blendshapes
            else []
        )

        # Initialize pose and hand processors
        self.pose_processor = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.3,
        )
        self.hand_processor = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="thirdparty/hand_landmarker.task"
                ),
                num_hands=2,
                min_hand_detection_confidence=0.5,
            )
        )

        # Initialize filters
        self.initialize_filters()

    def initialize_filters(self):
        # Body and face keypoints (45 total)
        num_keypoints = len(MEDIAPIPE_KEYPOINT_NAMES)
        self.keypoint_filters = [
            {
                "x": OneEuroFilter(30, 1.0, 0.7, 1.0),
                "y": OneEuroFilter(30, 1.0, 0.7, 1.0),
                "z": OneEuroFilter(30, 1.0, 0.7, 1.0),
                "vis": OneEuroFilter(30, 1.0, 0.7, 1.0),
            }
            for _ in range(num_keypoints)
        ]

        # Blendshapes
        self.blendshape_filters = {
            name: OneEuroFilter(30, 1.0, 0.7, 1.0) for name in self.blendshape_names
        }

        # Hands (21 landmarks per hand, 3 coordinates each)
        self.hand_filters = {
            "left": [
                {
                    "x": OneEuroFilter(30, 1.0, 0.7, 1.0),
                    "y": OneEuroFilter(30, 1.0, 0.7, 1.0),
                    "z": OneEuroFilter(30, 1.0, 0.7, 1.0),
                }
                for _ in range(21)
            ],
            "right": [
                {
                    "x": OneEuroFilter(30, 1.0, 0.7, 1.0),
                    "y": OneEuroFilter(30, 1.0, 0.7, 1.0),
                    "z": OneEuroFilter(30, 1.0, 0.7, 1.0),
                }
                for _ in range(21)
            ],
        }

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
            description="Process every nth frame for video input", ge=1, default=1
        ),
    ) -> Output:
        if str(media_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
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
                person_result.update({"person_id": person_id, "box": box})
                all_results.append(person_result)

        annotated_frame = self.annotate_video_frame(img_np, all_results)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            json.dump(
                {
                    "mediapipe_keypoints": self.aggregate_mediapipe(
                        all_results, original_w, original_h
                    ),
                    "blendshapes": {"people": [r["blendshapes"] for r in all_results]},
                    "fullbody_data": {"people": [r["fullbody"] for r in all_results]},
                    "hand_landmarks": [r["hands"] for r in all_results]
                    if any(r["hands"] for r in all_results)
                    else None,
                },
                tmp,
            )
            mediapipe_keypoints_path = Path(tmp.name)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            Image.fromarray(annotated_frame).save(tmp.name)
            debug_media_path = Path(tmp.name)

        return Output(
            mediapipe_keypoints=mediapipe_keypoints_path,
            blendshapes=mediapipe_keypoints_path,
            fullbody_data=mediapipe_keypoints_path,
            debug_media=debug_media_path,
            hand_landmarks=mediapipe_keypoints_path
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

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            debug_video_path = tmp.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            debug_video_path, fourcc, fps / frame_sample_rate, (width, height)
        )

        frame_results = []
        frame_count = 0
        processed_count = 0

        progress = tqdm(
            total=total_frames,
            desc="Processing video",
            unit="frame",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            progress.update(1)

            if frame_count % frame_sample_rate != 0:
                frame_count += 1
                continue

            img_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = PersonProcessor.detect_people(img_np, max_people)
            all_results = []

            for person_id, box in enumerate(boxes):
                startX, startY, endX, endY = box
                crop = img_np[startY:endY, startX:endX]
                if crop.size == 0:
                    continue

                person_result = PersonProcessor.process_crop(
                    crop, box, (height, width), self
                )
                if person_result:
                    person_result.update({"person_id": person_id, "box": box})
                    all_results.append(person_result)

            if all_results:
                main_person = all_results[0]
                timestamp = frame_count / fps if fps > 0 else 0
                self.apply_filters(main_person, timestamp)

            annotated_frame = self.annotate_video_frame(img_np, all_results)
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            frame_results.append(
                {
                    "mediapipe": self.aggregate_mediapipe(all_results, width, height),
                    "blendshapes": [r["blendshapes"] for r in all_results],
                    "fullbody": [r["fullbody"] for r in all_results],
                    "hands": [r["hands"] for r in all_results],
                    "num_people": len(all_results),
                }
            )

            processed_count += 1
            frame_count += 1
            progress.set_postfix_str(f"Processed: {processed_count} frames")

        progress.close()
        cap.release()
        out.release()

        # Add audio to the debug video
        original_video = VideoFileClip(str(video_path))
        debug_video = VideoFileClip(debug_video_path)
        final_video_with_audio = debug_video.with_audio(original_video.audio)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_final:
            debug_with_audio_path = tmp_final.name
        final_video_with_audio.write_videofile(debug_with_audio_path)

        # Convert the video to a widely supported format
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_final_conv:
            final_video_path = tmp_final_conv.name

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                debug_with_audio_path,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                final_video_path,
            ],
            check=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            json.dump(
                {
                    "mediapipe_keypoints": [f["mediapipe"] for f in frame_results],
                    "blendshapes": {
                        "frames": [{"people": f["blendshapes"]} for f in frame_results]
                    },
                    "fullbody_data": {
                        "frames": [{"people": f["fullbody"]} for f in frame_results]
                    },
                    "hand_landmarks": [f["hands"] for f in frame_results],
                },
                tmp,
            )
            mediapipe_keypoints_path = Path(tmp.name)

        return Output(
            mediapipe_keypoints=mediapipe_keypoints_path,
            blendshapes=mediapipe_keypoints_path,
            fullbody_data=mediapipe_keypoints_path,
            debug_media=Path(final_video_path),
            hand_landmarks=mediapipe_keypoints_path,
            num_people=max(f["num_people"] for f in frame_results),
            media_type="video",
            total_frames=processed_count,
        )

    def apply_filters(self, person_data, timestamp):
        # Filter keypoints
        for kp in person_data["fullbody"]["keypoints"]:
            kp_id = kp["id"]
            if kp_id >= len(self.keypoint_filters):
                continue

            x = kp["position"][0]
            y = kp["position"][1]
            z = kp["position"][2]
            vis = kp["visibility"]

            kp["position"] = [
                self.keypoint_filters[kp_id]["x"](x, timestamp),
                self.keypoint_filters[kp_id]["y"](y, timestamp),
                self.keypoint_filters[kp_id]["z"](z, timestamp),
            ]
            kp["visibility"] = self.keypoint_filters[kp_id]["vis"](vis, timestamp)

        # Filter blendshapes
        filtered_blendshapes = []
        for bs in person_data["blendshapes"]:
            name = bs["name"]
            if name in self.blendshape_filters:
                filtered_score = self.blendshape_filters[name](bs["score"], timestamp)
                filtered_blendshapes.append({"name": name, "score": filtered_score})
        person_data["blendshapes"] = filtered_blendshapes

        # Filter hands with error handling
        for hand_type in ["left", "right"]:
            hand = person_data["hands"].get(hand_type, [])
            for idx, landmark in enumerate(hand):
                if idx >= 21:
                    continue
                filters = self.hand_filters[hand_type][idx]

                # Safe coordinate access with defaults
                x = landmark.get("x", 0.0)
                y = landmark.get("y", 0.0)
                z = landmark.get("z", 0.0)

                # Apply filters
                landmark["x"] = filters["x"](x, timestamp)
                landmark["y"] = filters["y"](y, timestamp)
                landmark["z"] = filters["z"](z, timestamp)

    def annotate_video_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
        annotated = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated)
        colors = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "orange": (255, 165, 0),
        }

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()

        for result in results:
            startX, startY, endX, endY = result["box"]
            draw.rectangle(
                [(startX, startY), (endX, endY)], outline=colors["green"], width=2
            )
            draw.text(
                (startX, startY - 20),
                f"Person {result['person_id']}",
                fill=colors["green"],
                font=font,
            )

            keypoints = result["fullbody"]["keypoints"]
            self.draw_skeleton(draw, keypoints, colors)

        return np.array(annotated)

    def draw_skeleton(self, draw, keypoints, colors):
        kp_dict = {kp["id"]: kp for kp in keypoints}
        for connection in FullBodyProcessor.SKELETON_CONNECTIONS:
            if connection[0] in kp_dict and connection[1] in kp_dict:
                parent = kp_dict[connection[0]]
                child = kp_dict[connection[1]]
                if parent["visibility"] < 0.5 or child["visibility"] < 0.5:
                    continue
                x1, y1 = parent["position"][0], parent["position"][1]
                x2, y2 = child["position"][0], child["position"][1]
                draw.line([(x1, y1), (x2, y2)], fill=colors["orange"], width=2)

        for kp in keypoints:
            if kp["visibility"] < 0.5 or kp["id"] > 32:
                continue
            x, y = kp["position"][0], kp["position"][1]
            bbox = [(x - 4, y - 4), (x + 4, y + 4)]
            draw.ellipse(bbox, fill=colors["red"], outline=None)

    def aggregate_mediapipe(self, results, width, height):
        annotations = []
        for idx, res in enumerate(results):
            ann = res["mediapipe"]["annotations"][0].copy()
            ann["id"] = idx
            annotations.append(ann)

        return {
            "info": {
                "description": "MediaPipe Full Body Keypoints",
                "version": "1.0",
                "year": 2023,
                "contributor": "MediaPipe Processor",
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
            "categories": FullBodyProcessor.create_mediapipe_output(None, None, (0, 0))[
                "categories"
            ],
        }
