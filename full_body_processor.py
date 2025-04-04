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

from models import (
    MEDIAPIPE_KEYPOINT_NAMES, SKELETON_CONNECTIONS)

# Define Mediapipe hand mappings for left and right hands
LEFT_HAND_MAPPING = {
    0: "left_wrist",
    1: "left_thumb_cmc",
    2: "left_thumb_mcp",
    3: "left_thumb_ip",
    4: "left_thumb_tip",
    5: "left_index_mcp",
    6: "left_index_pip",
    7: "left_index_dip",
    8: "left_index_tip",
    9: "left_middle_mcp",
    10: "left_middle_pip",
    11: "left_middle_dip",
    12: "left_middle_tip",
    13: "left_ring_mcp",
    14: "left_ring_pip",
    15: "left_ring_dip",
    16: "left_ring_tip",
    17: "left_pinky_mcp",
    18: "left_pinky_pip",
    19: "left_pinky_dip",
    20: "left_pinky_tip",
}

RIGHT_HAND_MAPPING = {
    0: "right_wrist",
    1: "right_thumb_cmc",
    2: "right_thumb_mcp",
    3: "right_thumb_ip",
    4: "right_thumb_tip",
    5: "right_index_mcp",
    6: "right_index_pip",
    7: "right_index_dip",
    8: "right_index_tip",
    9: "right_middle_mcp",
    10: "right_middle_pip",
    11: "right_middle_dip",
    12: "right_middle_tip",
    13: "right_ring_mcp",
    14: "right_ring_pip",
    15: "right_ring_dip",
    16: "right_ring_tip",
    17: "right_pinky_mcp",
    18: "right_pinky_pip",
    19: "right_pinky_dip",
    20: "right_pinky_tip",
}

class FullBodyProcessor:
    @staticmethod
    def process_results(pose, left_hand, right_hand, image_size):
        return {
            "mediapipe": FullBodyProcessor.create_mediapipe_output(
                pose, image_size
            ),
            "fullbody": FullBodyProcessor.create_fullbody_output(
                pose, image_size
            ),
            "hands": FullBodyProcessor.process_hands(left_hand, right_hand),
        }

    @staticmethod
    def create_mediapipe_output(pose, image_size):
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
                    "skeleton": SKELETON_CONNECTIONS,
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
    def create_fullbody_output(pose, image_size):
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
                        LEFT_HAND_MAPPING if is_left else RIGHT_HAND_MAPPING
                    ).get(idx, None),
                }
                for idx, lmk in enumerate(hand)
            ]

        return {
            "left": process_single_hand(left_hand, True),
            "right": process_single_hand(right_hand, False),
        }

    def process_combined_keypoints(self, crop):
        pose_result = self.pose_processor.process(crop)
        body_keypoints = extract_body_keypoints(pose_result)

        hand_result = self.hand_processor.detect(crop)
        left_hand, right_hand = extract_hand_keypoints(hand_result)

        combined_keypoints = body_keypoints + left_hand + right_hand

        while len(combined_keypoints) < 132:
            combined_keypoints.extend([0.0, 0.0, 0])
        combined_keypoints = combined_keypoints[:132]  # Truncate if longer

        return {"keypoints": combined_keypoints}

def extract_body_keypoints(pose_result):
    """
    Extracts body keypoints from the pose result.
    """
    if not pose_result or not pose_result.pose_landmarks:
        return [0.0, 0.0, 0] * 33  # Return 33 keypoints with default values

    keypoints = []
    for lmk in pose_result.pose_landmarks.landmark[:33]:
        keypoints.extend([lmk.x, lmk.y, lmk.visibility])
    return keypoints

def extract_hand_keypoints(hand_result):
    """
    Extracts left and right hand keypoints from the hand result.
    """
    if not hand_result:
        return ([0.0, 0.0, 0] * 21, [0.0, 0.0, 0] * 21)  # Default 21 keypoints per hand

    left_hand = []
    right_hand = []

    if hand_result.multi_hand_landmarks:
        for hand_landmarks, hand_label in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
            keypoints = []
            for lmk in hand_landmarks.landmark:
                keypoints.extend([lmk.x, lmk.y, lmk.z])

            if hand_label.classification[0].label.lower() == "left":
                left_hand = keypoints
            else:
                right_hand = keypoints

    # Ensure both hands have 21 keypoints
    if not left_hand:
        left_hand = [0.0, 0.0, 0] * 21
    if not right_hand:
        right_hand = [0.0, 0.0, 0] * 21

    return left_hand, right_hand