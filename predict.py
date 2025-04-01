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
import numpy as np
import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import Category, Landmark
from cog import BasePredictor, Input, Path, BaseModel
from PIL import Image
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
from collections import deque

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
COCO_SKELETON = [
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
    [0, 17],
    [0, 18],
    [17, 19],
    [18, 20],
    [17, 21],
    [18, 22],
    [21, 23],
    [22, 24],
    [0, 25],
    [0, 26],
    [25, 27],
    [26, 28],
]
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


class Output(BaseModel):
    coco_keypoints: str
    facs: str
    fullbodyfacs: str
    debug_image: Path
    hand_landmarks: Optional[str]


class FullBodyProcessor:
    @staticmethod
    def process_results(
        pose_landmarks: Optional[landmark_pb2.NormalizedLandmarkList],
        face_landmarks: Optional[landmark_pb2.NormalizedLandmarkList],
        face_blendshapes: List[Any],
        left_hand: Optional[List[Landmark]],
        right_hand: Optional[List[Landmark]],
        image_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        return {
            "coco": FullBodyProcessor._create_coco_output(
                pose_landmarks, face_landmarks, image_size
            ),
            "facs": FullBodyProcessor._create_facs_output(face_blendshapes),
            "fullbodyfacs": FullBodyProcessor._create_fullbodyfacs(
                pose_landmarks, face_landmarks, image_size
            ),
        }

    @staticmethod
    def _create_coco_output(pose, face, image_size):
        """Generate COCO 1.1 compliant JSON output"""
        height, width = image_size
        keypoints = []
        num_visible = 0

        body_landmarks = pose.landmark if pose else []
        for idx in range(17):
            if idx < len(body_landmarks):
                lmk = body_landmarks[idx]
                x, y, vis = lmk.x * width, lmk.y * height, lmk.visibility
                keypoints += [x, y, 2 if vis > 0.5 else 1]
                num_visible += 1 if vis > 0 else 0
            else:
                keypoints += [0.0, 0.0, 0]
        face_landmarks_list = face if face else []
        facial_indices = [151, 334, 46, 276, 159, 386, 145, 374, 13, 14, 61, 291]
        for idx in facial_indices:
            if idx < len(face_landmarks_list):
                lmk = face_landmarks_list[idx]
                keypoints += [lmk.x * width, lmk.y * height, 2]
                num_visible += 1
            else:
                keypoints += [0.0, 0.0, 0]
        return {
            "info": {
                "description": "COCO 1.1 Extended with Facial Keypoints",
                "version": "1.1",
                "year": 2023,
                "contributor": "MediaPipe FACS Extension",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "CC-BY-4.0",
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                }
            ],
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
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": num_visible,
                    "bbox": FullBodyProcessor._calculate_bbox(keypoints),
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
    def _create_facs_output(blendshapes: List[Any]) -> Dict[str, Any]:
        """Generate FACS-compliant AU intensities"""
        au_scores = {}
        blendshape_dict = {}
        for bs in blendshapes:
            if hasattr(bs, "category_name") and hasattr(bs, "score"):
                blendshape_dict[bs.category_name] = bs.score

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
    def _create_fullbodyfacs(pose, face, image_size):
        """Generate FullBodyFACS hierarchy"""
        keypoints = []
        height, width = image_size
        body_landmarks = pose.landmark if pose else []
        for idx in range(33):
            if idx < len(body_landmarks):
                lmk = body_landmarks[idx]
                keypoints.append(
                    {
                        "id": idx,
                        "name": f"body_{idx}",
                        "position": [lmk.x * width, lmk.y * height, lmk.z * width],
                        "parent": FullBodyProcessor._get_parent(idx),
                    }
                )
        facial_map = {
            151: 17,
            334: 18,
            46: 19,
            276: 20,
            159: 21,
            386: 22,
            145: 23,
            374: 24,
            13: 25,
            14: 26,
            61: 27,
            291: 28,
        }
        face_landmarks_list = face if face else []
        for mp_idx, facs_idx in facial_map.items():
            if mp_idx < len(face_landmarks_list):
                lmk = face_landmarks_list[mp_idx]
                keypoints.append(
                    {
                        "id": facs_idx,
                        "name": COCO_KEYPOINT_NAMES[facs_idx],
                        "position": [lmk.x * width, lmk.y * height, lmk.z * width],
                        "parent": 0,
                    }
                )
        return {"keypoints": keypoints}

    @staticmethod
    def _get_parent(idx: int) -> int:
        hierarchy = {
            0: -1,
            11: 5,
            12: 6,
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
        }
        return hierarchy.get(idx, -1)

    @staticmethod
    def _calculate_bbox(keypoints: List[float]) -> List[float]:
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


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Initialize MediaPipe models"""
        self.face_processor = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="thirdparty/face_landmarker_v2_with_blendshapes.task"
                ),
                output_face_blendshapes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )
        self.pose_processor = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            smooth_landmarks=True
        )
        self.hand_processor = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="thirdparty/hand_landmarker.task"
                ),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

    def predict(self, image_path: Path = Input(description="Input image")) -> Output:
        """Process an image and return all outputs"""
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        height, width, _ = img_np.shape
        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=img_np.astype(np.uint8)
        )
        # Detect features
        face_result = self.face_processor.detect(mp_image)
        pose_result = self.pose_processor.process(img_np)
        hand_result = self.hand_processor.detect(mp_image)
        # Process hands
        left_hand, right_hand = None, None
        if hand_result.hand_landmarks:
            for idx, handedness in enumerate(hand_result.handedness):
                if handedness[0].display_name == "Left":
                    left_hand = hand_result.hand_landmarks[idx]
                else:
                    right_hand = hand_result.hand_landmarks[idx]
        # Process results
        results = FullBodyProcessor.process_results(
            pose_landmarks=pose_result.pose_landmarks,
            face_landmarks=face_result.face_landmarks[0]
            if face_result.face_landmarks
            else None,
            face_blendshapes=face_result.face_blendshapes[0]
            if face_result.face_blendshapes
            else [],
            left_hand=left_hand,
            right_hand=right_hand,
            image_size=(height, width),
        )
        return Output(
            coco_keypoints=json.dumps(results["coco"], indent=2),
            facs=json.dumps(results["facs"], indent=2),
            fullbodyfacs=json.dumps(results["fullbodyfacs"], indent=2),
            debug_image=self._create_debug_image(
                img_np, face_result, pose_result, hand_result, image_path
            ),
            hand_landmarks=json.dumps(
                {
                    "left": [{"x": lmk.x, "y": lmk.y} for lmk in left_hand]
                    if left_hand
                    else [],
                    "right": [{"x": lmk.x, "y": lmk.y} for lmk in right_hand]
                    if right_hand
                    else [],
                }
            )
            if left_hand or right_hand
            else None,
        )

    def _create_debug_image(
        self,
        img_np: np.ndarray,
        face_result: vision.FaceLandmarkerResult,
        pose_result: mp.tasks.vision.PoseLandmarkerResult,
        hand_result: vision.HandLandmarkerResult,
        image_path: Path,
    ) -> Path:
        """Generate annotated debug image"""
        annotated = img_np.copy()
        # Draw face landmarks
        if face_result.face_landmarks:
            landmarks = landmark_pb2.NormalizedLandmarkList()
            landmarks.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=lmk.x, y=lmk.y, z=lmk.z)
                    for lmk in face_result.face_landmarks[0]
                ]
            )
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS
            )
        # Draw pose skeleton
        if pose_result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated,
                pose_result.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
            )
        # Draw hand landmarks
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(x=lmk.x, y=lmk.y, z=lmk.z)
                        for lmk in hand_landmarks
                    ]
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated, hand_landmarks_proto, mp.solutions.hands.HAND_CONNECTIONS
                )
        debug_path = f"/tmp/{os.path.basename(image_path)}_debug.jpg"
        Image.fromarray(annotated).save(debug_path)
        return Path(debug_path)


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    result = predictor.predict(image_path)
    print(json.dumps(json.loads(result.json()), indent=2))
