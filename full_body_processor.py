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
                    "z": lmk.z,
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