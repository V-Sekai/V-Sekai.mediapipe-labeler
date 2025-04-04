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

from cog import BaseModel
from typing import Optional
from pathlib import Path

MEDIAPIPE_KEYPOINT_NAMES = [
    # Body (33 keypoints)
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
    # Left hand (21 keypoints)
    "left_wrist", "left_thumb_cmc", "left_thumb_mcp", "left_thumb_ip", "left_thumb_tip", "left_index_mcp", "left_index_pip", "left_index_dip", "left_index_tip", "left_middle_mcp", "left_middle_pip", "left_middle_dip", "left_middle_tip", "left_ring_mcp", "left_ring_pip", "left_ring_dip", "left_ring_tip", "left_pinky_mcp", "left_pinky_pip", "left_pinky_dip", "left_pinky_tip",
    # Right hand (21 keypoints)
    "right_wrist", "right_thumb_cmc", "right_thumb_mcp", "right_thumb_ip", "right_thumb_tip", "right_index_mcp", "right_index_pip", "right_index_dip", "right_index_tip", "right_middle_mcp", "right_middle_pip", "right_middle_dip", "right_middle_tip", "right_ring_mcp", "right_ring_pip", "right_ring_dip", "right_ring_tip", "right_pinky_mcp", "right_pinky_pip", "right_pinky_dip", "right_pinky_tip"
]

SKELETON_CONNECTIONS = [
    # Body connections
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10], [11, 12], [12, 14], [14, 16], [11, 13], [13, 15], [15, 17],
    [12, 24], [24, 26], [26, 28], [11, 23], [23, 25], [25, 27],
    [28, 30], [30, 32], [27, 29], [29, 31],
    # Left hand connections
    [33, 34], [34, 35], [35, 36], [36, 37],  # Thumb
    [33, 38], [38, 39], [39, 40], [40, 41],  # Index finger
    [33, 42], [42, 43], [43, 44], [44, 45],  # Middle finger
    [33, 46], [46, 47], [47, 48], [48, 49],  # Ring finger
    [33, 50], [50, 51], [51, 52], [52, 53],  # Pinky
    # Right hand connections
    [54, 55], [55, 56], [56, 57], [57, 58],  # Thumb
    [54, 59], [59, 60], [60, 61], [61, 62],  # Index finger
    [54, 63], [63, 64], [64, 65], [65, 66],  # Middle finger
    [54, 67], [67, 68], [68, 69], [69, 70],  # Ring finger
    [54, 71], [71, 72], [72, 73], [73, 74]   # Pinky
]

class Output(BaseModel):
    annotations: str
    debug_media: Path
    num_people: int
    media_type: str
    export_train_folder: Optional[Path] = None
    export_aligned_train_folder: Optional[Path] = None

