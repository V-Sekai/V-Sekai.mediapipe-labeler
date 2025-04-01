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
from PIL import ImageDraw, ImageFont
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


class SkeletalValidator:
    BONE_RATIOS = {
        ("left_shoulder", "left_elbow", "left_wrist"): (0.6, 0.3),
        ("right_shoulder", "right_elbow", "right_wrist"): (0.6, 0.3),
        ("left_hip", "left_knee", "left_ankle"): (0.55, 0.25),
        ("right_hip", "right_knee", "right_ankle"): (0.55, 0.25),
        ("nose", "neck", "mid_hip"): (0.8, 0.3),
    }

    JOINT_ANGLE_RANGES = {
        "elbow": (20, 160),
        "knee": (50, 180),
        "shoulder": (60, 180),
        "hip": (70, 120),
    }

    @classmethod
    def validate_skeleton(cls, keypoints):
        try:
            kp_dict = {kp["name"]: kp for kp in keypoints}

            if not cls.check_parent_connections(kp_dict):
                return False
            if not cls.validate_bone_ratios(kp_dict):
                return False
            if not cls.validate_joint_angles(kp_dict):
                return False
            if not cls.validate_global_proportions(kp_dict):
                return False

            return True
        except KeyError:
            return False

    @classmethod
    def check_parent_connections(cls, kp_dict):
        hierarchy = {
            "nose": ["neck"],
            "neck": ["left_shoulder", "right_shoulder", "mid_hip"],
            "left_shoulder": ["left_elbow"],
            "left_elbow": ["left_wrist"],
            "right_shoulder": ["right_elbow"],
            "right_elbow": ["right_wrist"],
            "mid_hip": ["left_hip", "right_hip"],
            "left_hip": ["left_knee"],
            "left_knee": ["left_ankle"],
            "right_hip": ["right_knee"],
            "right_knee": ["right_ankle"],
        }

        for parent, children in hierarchy.items():
            for child in children:
                if kp_dict[child]["parent"] != parent:
                    return False
        return True

    @classmethod
    def validate_bone_ratios(cls, kp_dict):
        for (a, b, c), (expected, tolerance) in cls.BONE_RATIOS.items():
            bone1 = cls.calculate_distance(kp_dict[a], kp_dict[b])
            bone2 = cls.calculate_distance(kp_dict[b], kp_dict[c])

            if bone1 == 0 or bone2 == 0:
                continue

            ratio = bone1 / bone2
            if not (expected - tolerance < ratio < expected + tolerance):
                return False
        return True

    @classmethod
    def validate_joint_angles(cls, kp_dict):
        angle_checks = {
            "elbow": [
                ("left_shoulder", "left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow", "right_wrist"),
            ],
            "knee": [
                ("left_hip", "left_knee", "left_ankle"),
                ("right_hip", "right_knee", "right_ankle"),
            ],
            "shoulder": [
                ("neck", "left_shoulder", "left_elbow"),
                ("neck", "right_shoulder", "right_elbow"),
            ],
            "hip": [
                ("mid_hip", "left_hip", "left_knee"),
                ("mid_hip", "right_hip", "right_knee"),
            ],
        }

        for joint_type, triples in angle_checks.items():
            min_angle, max_angle = cls.JOINT_ANGLE_RANGES[joint_type]
            for a, b, c in triples:
                angle = cls.calculate_angle(kp_dict[a], kp_dict[b], kp_dict[c])
                if not (min_angle <= angle <= max_angle):
                    return False
        return True

    @classmethod
    def validate_global_proportions(cls, kp_dict):
        try:
            arm_span = cls.calculate_distance(
                kp_dict["left_wrist"], kp_dict["right_wrist"]
            )
            height = cls.calculate_distance(
                kp_dict["nose"], kp_dict["mid_hip"]
            ) + cls.calculate_distance(kp_dict["mid_hip"], kp_dict["left_ankle"])
            return 0.7 < arm_span / height < 1.3
        except KeyError:
            return False

    @staticmethod
    def calculate_distance(a, b):
        return np.hypot(
            a["position"][0] - b["position"][0], a["position"][1] - b["position"][1]
        )

    @staticmethod
    def calculate_angle(a, b, c):
        ba = np.array(
            [a["position"][0] - b["position"][0], a["position"][1] - b["position"][1]]
        )
        bc = np.array(
            [c["position"][0] - b["position"][0], c["position"][1] - b["position"][1]]
        )
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))


class Output(BaseModel):
    coco_keypoints: str
    facs: str
    fullbodyfacs: str
    debug_image: Path
    hand_landmarks: Optional[str]
    num_people: int


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

            # Expand box by 20% with boundary checks
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

        # Coordinate mapping functions
        def map_x(x):
            return (x / new_w) * (endX - startX) + startX

        def map_y(y):
            return (y / new_h) * (endY - startY) + startY

        def map_z(z):
            return z * scale

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)
            face_result = predictor.face_processor.detect(mp_image)
            pose_result = predictor.pose_processor.process(resized)
            hand_result = predictor.hand_processor.detect(mp_image)
        except Exception as e:
            print(f"Processing error: {e}")
            return None

        # Process pose landmarks
        mapped_pose = None
        if pose_result.pose_landmarks:
            mapped_pose = landmark_pb2.NormalizedLandmarkList()
            for lmk in pose_result.pose_landmarks.landmark:
                mapped_pose.landmark.append(
                    landmark_pb2.NormalizedLandmark(
                        x=map_x(lmk.x * new_w) / orig_w,
                        y=map_y(lmk.y * new_h) / orig_h,
                        z=map_z(lmk.z),
                        visibility=lmk.visibility,
                    )
                )

        # Process face landmarks
        mapped_face = []
        if face_result.face_landmarks:
            for lmk in face_result.face_landmarks[0]:
                mapped_face.append(
                    Landmark(
                        x=map_x(lmk.x * new_w) / orig_w,
                        y=map_y(lmk.y * new_h) / orig_h,
                        z=map_z(lmk.z),
                    )
                )

        # Process hands
        left_hand, right_hand = [], []
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

        return FullBodyProcessor.process_results(
            mapped_pose,
            mapped_face,
            face_result.face_blendshapes[0] if face_result.face_blendshapes else [],
            left_hand,
            right_hand,
            original_size,
        )


class FullBodyProcessor:
    @staticmethod
    def process_results(pose, face, blendshapes, left_hand, right_hand, image_size):
        result = {
            "coco": FullBodyProcessor.create_coco_output(pose, face, image_size),
            "facs": FullBodyProcessor.create_facs_output(blendshapes),
            "fullbodyfacs": FullBodyProcessor.create_fullbodyfacs(
                pose, face, image_size
            ),
            "hands": FullBodyProcessor.process_hands(left_hand, right_hand),
        }
        result["valid"] = SkeletalValidator.validate_skeleton(
            result["fullbodyfacs"]["keypoints"]
        )
        return result

    @staticmethod
    def create_coco_output(pose, face, image_size):
        height, width = image_size
        keypoints = []
        num_visible = 0

        if pose:
            for idx in range(17):
                if idx < len(pose.landmark):
                    lmk = pose.landmark[idx]
                    x, y, vis = lmk.x * width, lmk.y * height, lmk.visibility
                    keypoints += [x, y, 2 if vis > 0.5 else 1]
                    num_visible += 1 if vis > 0 else 0
                else:
                    keypoints += [0.0, 0.0, 0]

        facial_indices = [151, 334, 46, 276, 159, 386, 145, 374, 13, 14, 61, 291]
        if face:
            for idx in facial_indices:
                if idx < len(face):
                    lmk = face[idx]
                    keypoints += [lmk.x * width, lmk.y * height, 2]
                    num_visible += 1
                else:
                    keypoints += [0.0, 0.0, 0]

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

        blendshape_dict = {
            bs.category_name: bs.score
            for bs in blendshapes
            if hasattr(bs, "category_name") and hasattr(bs, "score")
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
                    }
                )

                if idx == 11:
                    left_shoulder = lmk
                elif idx == 12:
                    right_shoulder = lmk

            if left_shoulder and right_shoulder:
                neck_x = (left_shoulder.x + right_shoulder.x) / 2 * width
                neck_y = (left_shoulder.y + right_shoulder.y) / 2 * height
                neck_z = (left_shoulder.z + right_shoulder.z) / 2 * width
                keypoints.append(
                    {
                        "id": 33,
                        "name": "neck",
                        "position": [neck_x, neck_y, neck_z],
                        "parent": 0,
                    }
                )

        facial_map = {
            151: 34,
            334: 35,
            46: 36,
            276: 37,
            159: 38,
            386: 39,
            145: 40,
            374: 41,
            13: 42,
            14: 43,
            61: 44,
            291: 45,
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
            return (
                [
                    {
                        "index": idx,
                        "x": lmk.x,
                        "y": lmk.y,
                        "name": ("left_wrist" if is_left else "right_wrist")
                        if idx == 0
                        else (
                            LEFT_HAND_VRM_MAPPING if is_left else RIGHT_HAND_VRM_MAPPING
                        ).get(idx),
                    }
                    for idx, lmk in enumerate(hand)
                ]
                if hand
                else []
            )

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
                    model_asset_path="thirdparty/face_landmarker.task"
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
                    model_asset_path="thirdparty/hand_landmarker.task"
                ),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    def predict(
        self,
        image_path: Path = Input(description="Input image"),
        max_people: int = Input(
            description="Maximum number of people to detect (1-100)",
            ge=1,
            le=100,
            default=100,
        ),
    ) -> Output:
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

        validated_results = []
        for result in all_results:
            if result.get("valid", False):
                validated_results.append(result)
            else:
                validated_results.append(self.create_fallback_result(result))

        return Output(
            coco_keypoints=json.dumps(
                self.aggregate_coco(validated_results, original_w, original_h), indent=2
            ),
            facs=json.dumps(
                {"people": [r["facs"] for r in validated_results]}, indent=2
            ),
            fullbodyfacs=json.dumps(
                {"people": [r["fullbodyfacs"] for r in validated_results]}, indent=2
            ),
            debug_image=self.create_debug_image(img_np, validated_results),
            hand_landmarks=json.dumps([r["hands"] for r in validated_results])
            if any(r["hands"] for r in validated_results)
            else None,
            num_people=len(validated_results),
        )

    def create_fallback_result(self, original_result):
        return {
            "coco": original_result["coco"],
            "facs": {"AUs": {}, "blendshapes": []},
            "fullbodyfacs": {
                "keypoints": [
                    kp
                    for kp in original_result["fullbodyfacs"]["keypoints"]
                    if kp["name"]
                    in [
                        "nose",
                        "left_shoulder",
                        "right_shoulder",
                        "left_hip",
                        "right_hip",
                        "mid_hip",
                    ]
                ]
            },
            "hands": {"left": [], "right": []},
            "valid": False,
            "box": original_result["box"],  # Add missing box information
            "person_id": original_result["person_id"],  # Add person_id
        }

    def create_debug_image(self, img_np: np.ndarray, results: list) -> Path:
        annotated = Image.fromarray(img_np)
        draw = ImageDraw.Draw(annotated)
        colors = {"valid": (0, 255, 0), "invalid": (255, 0, 0)}

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()

        for result in results:
            box = result["box"]
            color = colors["valid"] if result.get("valid", False) else colors["invalid"]

            # Draw bounding box
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=2)

            # Draw keypoints
            for kp in result["fullbodyfacs"]["keypoints"]:
                x, y = kp["position"][0], kp["position"][1]
                bbox = [(x - 4, y - 4), (x + 4, y + 4)]
                draw.ellipse(bbox, fill=color, outline=None)

            # Draw person ID
            label = f"Person {result['person_id']}"
            draw.text((box[0], box[1] - 20), label, fill=color, font=font)

        debug_path = "/tmp/debug_output.jpg"
        annotated.save(debug_path)
        return Path(debug_path)

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
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    result = predictor.predict(image_path)
    print(json.dumps(json.loads(result.json()), indent=2))
