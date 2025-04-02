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
import subprocess
import json
from PIL import ImageDraw
import math
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from cog import BasePredictor, Input, Path, BaseModel
from PIL import Image
from typing import List, Tuple, Optional
from datetime import datetime


# 1€ Filter Implementation
class LowPassFilter:
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


# Configuration
COCO_KEYPOINT_NAMES = [
    # Original 29 body/face keypoints
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
    # Left hand (21 keypoints)
    *[f"left_hand_{i}" for i in range(21)],
    # Right hand (21 keypoints)
    *[f"right_hand_{i}" for i in range(21)],
]

COCO_SKELETON = [
    # Original body connections
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
    # Hand connections (optional)
    # Left hand
    [29, 30],
    [30, 31],
    [31, 32],
    [32, 33],  # Thumb
    [29, 34],
    [34, 35],
    [35, 36],
    [36, 37],  # Index
    [29, 38],
    [38, 39],
    [39, 40],
    [40, 41],  # Middle
    [29, 42],
    [42, 43],
    [43, 44],
    [44, 45],  # Ring
    [29, 46],
    [46, 47],
    [47, 48],
    [48, 49],  # Pinky
    # Right hand
    [50, 51],
    [51, 52],
    [52, 53],
    [53, 54],  # Thumb
    [50, 55],
    [55, 56],
    [56, 57],
    [57, 58],  # Index
    [50, 59],
    [59, 60],
    [60, 61],
    [61, 62],  # Middle
    [50, 63],
    [63, 64],
    [64, 65],
    [65, 66],  # Ring
    [50, 67],
    [67, 68],
    [68, 69],
    [69, 70],  # Pinky
]


class Output(BaseModel):
    coco_keypoints: str
    blendshapes: str
    debug_media: Path
    num_people: int
    media_type: str
    total_frames: Optional[int] = None


class PersonProcessor:
    @staticmethod
    def detect_people(
        image: np.ndarray, max_people: int
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
        predictor,
    ) -> Optional[dict]:
        (startX, startY, endX, endY), (orig_h, orig_w) = box, original_size
        h, w = crop.shape[:2]
        scale = 640 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (new_w, new_h))

        try:
            face_result = predictor.face_processor.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)
            )
            pose_result = predictor.pose_processor.process(resized)
            hand_result = predictor.hand_processor.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)
            )
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

        # Process pose landmarks
        mapped_pose = landmark_pb2.NormalizedLandmarkList()
        if pose_result.pose_landmarks:
            for lmk in pose_result.pose_landmarks.landmark:
                if lmk.visibility < 0.1:
                    mapped_pose.landmark.add(x=0, y=0, z=0, visibility=0)
                else:
                    mapped_pose.landmark.add(
                        x=map_x(lmk.x * new_w) / orig_w,
                        y=map_y(lmk.y * new_h) / orig_h,
                        z=map_z(lmk.z),
                        visibility=lmk.visibility,
                    )
        # Process face landmarks
        mapped_face = []
        if face_result.face_landmarks:
            facial_indices = [105, 334, 46, 276, 159, 386, 145, 374, 13, 14, 61, 291]
            for idx in facial_indices:
                if idx < len(face_result.face_landmarks[0]):
                    lmk = face_result.face_landmarks[0][idx]
                    mapped_face.append(
                        {
                            "x": float(map_x(lmk.x * new_w) / orig_w),
                            "y": float(map_y(lmk.y * new_h) / orig_h),
                            "z": float(map_z(lmk.z)),
                        }
                    )
        # Process hands
        left_hand, right_hand = [], []
        if hand_result.hand_landmarks:
            for idx, handedness in enumerate(hand_result.handedness):
                hand = [
                    {
                        "x": float(map_x(lmk.x * new_w) / orig_w),
                        "y": float(map_y(lmk.y * new_h) / orig_h),
                        "z": float(map_z(lmk.z)),
                    }
                    for lmk in hand_result.hand_landmarks[idx]
                ]

                if handedness[0].display_name == "Left":
                    left_hand = hand
                else:
                    right_hand = hand
        return {
            "coco": FullBodyProcessor.create_coco_output(
                mapped_pose, mapped_face, left_hand, right_hand, original_size
            ),
            "blendshapes": [
                {"name": bs.category_name, "score": float(bs.score)}
                for bs in face_result.face_blendshapes[0]
            ]
            if face_result.face_blendshapes
            else [],
        }


class FullBodyProcessor:
    @staticmethod
    def create_coco_output(pose, face, left_hand, right_hand, image_size) -> dict:
        height, width = image_size
        keypoints = []
        num_visible = 0
        # Process body keypoints
        if pose:
            for idx in range(17):
                if idx < len(pose.landmark):
                    lmk = pose.landmark[idx]
                    if lmk.visibility >= 0.1:
                        vis_flag = 2 if lmk.visibility > 0.5 else 1
                        keypoints += [lmk.x * width, lmk.y * height, vis_flag]
                        num_visible += 1 if vis_flag == 2 else 0
                    else:
                        keypoints += [0.0, 0.0, 0]
                else:
                    keypoints += [0.0, 0.0, 0]
        # Process facial keypoints
        for lmk in face:
            keypoints += [lmk["x"] * width, lmk["y"] * height, 2]
            num_visible += 1
        # Process hands (21 landmarks per hand)
        for lmk in left_hand:
            keypoints += [lmk["x"] * width, lmk["y"] * height, 2]
            num_visible += 1
        for lmk in right_hand:
            keypoints += [lmk["x"] * width, lmk["y"] * height, 2]
            num_visible += 1
        # Neck position adjustment
        if pose and len(pose.landmark) > 12:
            nose = pose.landmark[0]
            left_sh = pose.landmark[11]
            right_sh = pose.landmark[12]
            if left_sh.visibility >= 0.5 and right_sh.visibility >= 0.5:
                neck_x = (left_sh.x + right_sh.x) / 2 * width
                keypoints[0] = (nose.x * width + neck_x) / 2
            keypoints[1] = nose.y * height
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
                    "skeleton": COCO_SKELETON,
                }
            ],
        }

    @staticmethod
    def calculate_bbox(keypoints):
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
                urllib.request.urlretrieve(url, path)
        # Initialize models
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

        self.pose_processor = mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=2, min_detection_confidence=0.5
        )

        self.hand_processor = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="thirdparty/hand_landmarker.task"
                ),
                num_hands=2,
            )
        )
        # Initialize filters
        self.filters = [OneEuroFilter() for _ in COCO_KEYPOINT_NAMES]

    def predict(
        self,
        media_path: Path,
        max_people: int = 20,
        frame_sample_rate: int = 1,
        max_processing_seconds: int = Input(
            description="Maximum processing time in seconds (0=unlimited)",
            ge=0,
            default=0,
        ),
    ) -> Output:
        if media_path.suffix.lower() in (".mp4", ".avi", ".mov"):
            return self.process_video(
                media_path, max_people, frame_sample_rate, max_processing_seconds
            )
        return self.process_image(media_path, max_people)

    def process_image(self, image_path: Path, max_people: int) -> Output:
        img = np.array(Image.open(image_path).convert("RGB"))
        h, w = img.shape[:2]
        results = []

        for box in PersonProcessor.detect_people(img, max_people):
            crop = img[box[1] : box[3], box[0] : box[2]]
            if crop.size == 0:
                continue
            result = PersonProcessor.process_crop(crop, box, (h, w), self)
            if result:
                results.append(result)
        return Output(
            coco_keypoints=json.dumps(self.aggregate_coco(results, w, h)),
            blendshapes=json.dumps({"people": [r["blendshapes"] for r in results]}),
            debug_media=self.create_debug_image(img, results),
            num_people=len(results),
            media_type="image",
        )

    def process_video(
        self,
        video_path: Path,
        max_people: int,
        frame_sample_rate: int,
        max_processing_seconds: int = 60,
    ) -> Output:
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        debug_video = cv2.VideoWriter(
            "/tmp/debug.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS) / frame_sample_rate,
            (width, height),
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
                results = []
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for box in PersonProcessor.detect_people(frame_rgb, max_people):
                    crop = frame_rgb[box[1] : box[3], box[0] : box[2]]
                    result = PersonProcessor.process_crop(
                        crop, box, (height, width), self
                    )
                    if result:
                        results.append(result)
                max_people_count = max(max_people_count, len(results))
                frame_results.append(
                    {
                        "coco": self.aggregate_coco(results, width, height),
                        "blendshapes": [r["blendshapes"] for r in results],
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
            blendshapes=json.dumps(
                {"frames": [{"people": f["blendshapes"]} for f in frame_results]}
            ),
            debug_media=Path("/tmp/debug.mp4"),
            num_people=max_people_count,
            media_type="video",
            total_frames=len(frame_results),
        )

    def annotate_frame(self, frame: np.ndarray, results: list) -> Image.Image:
        annotated = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated)
        colors = {
            "body": (0, 255, 0),  # Green
            "face": (255, 0, 0),  # Red
            "left_hand": (0, 0, 255),  # Blue
            "right_hand": (255, 0, 255),  # Magenta
        }
        for result in results:
            box = result.get("box", (0, 0, 0, 0))
            draw.rectangle(
                [(box[0], box[1]), (box[2], box[3])], outline=colors["body"], width=2
            )
            keypoints = result["coco"]["annotations"][0]["keypoints"]
            num_keypoints = len(keypoints) // 3

            # Validate keypoint count
            if num_keypoints != len(COCO_KEYPOINT_NAMES):
                print(
                    f"Warning: Keypoint count mismatch. Expected {len(COCO_KEYPOINT_NAMES)}, got {num_keypoints}"
                )
            # Draw keypoints with color coding
            for i in range(num_keypoints):
                x = keypoints[i * 3]
                y = keypoints[i * 3 + 1]
                vis = keypoints[i * 3 + 2]
                if vis > 0:
                    color = colors["body"]
                    if 17 <= i <= 28:  # Face
                        color = colors["face"]
                    elif 29 <= i <= 49:  # Left hand
                        color = colors["left_hand"]
                    elif 50 <= i <= 70:  # Right hand
                        color = colors["right_hand"]
                    draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=color)
            # Draw skeleton connections with validation
            for a, b in COCO_SKELETON:
                if a >= num_keypoints or b >= num_keypoints:
                    continue  # Skip invalid connections
                ax = keypoints[a * 3]
                ay = keypoints[a * 3 + 1]
                avis = keypoints[a * 3 + 2]
                bx = keypoints[b * 3]
                by = keypoints[b * 3 + 1]
                bvis = keypoints[b * 3 + 2]

                if avis > 0 and bvis > 0:
                    color = colors["body"]
                    # Determine connection group
                    if (29 <= a <= 49) or (29 <= b <= 49):
                        color = colors["left_hand"]
                    elif (50 <= a <= 70) or (50 <= b <= 70):
                        color = colors["right_hand"]
                    elif (17 <= a <= 28) or (17 <= b <= 28):
                        color = colors["face"]

                    draw.line([(ax, ay), (bx, by)], fill=color, width=2)
        return annotated

    def create_debug_image(self, img: np.ndarray, results: list) -> Path:
        pil_img = Image.fromarray(img)
        debug_img = self.annotate_frame(np.array(pil_img), results)
        debug_img.save("/tmp/debug.jpg")
        return Path("/tmp/debug.jpg")

    def aggregate_coco(self, results, width, height):
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
                    "skeleton": COCO_SKELETON,
                }
            ],
        }
