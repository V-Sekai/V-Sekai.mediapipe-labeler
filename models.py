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
    # Left Hand (11 keypoints)
    "left_wrist", "left_thumb_tip", "left_index_tip", "left_middle_tip", "left_ring_tip", "left_pinky_tip", "left_thumb_ip", "left_index_dip", "left_middle_dip", "left_ring_dip", "left_pinky_dip",
    # Right Hand (11 keypoints)
    "right_wrist", "right_thumb_tip", "right_index_tip", "right_middle_tip", "right_ring_tip", "right_pinky_tip", "right_thumb_ip", "right_index_dip", "right_middle_dip", "right_ring_dip", "right_pinky_dip"
]

class Output(BaseModel):
    annotations: str
    debug_media: Path
    num_people: int
    media_type: str
    export_train_folder: Optional[Path] = None
    export_aligned_train_folder: Optional[Path] = None

