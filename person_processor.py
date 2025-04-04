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
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import Landmark
from typing import List, Tuple, Any
import mediapipe as mp
from mediapipe.tasks.python.vision import ObjectDetector, ObjectDetectorOptions
from full_body_processor import FullBodyProcessor
from mediapipe.tasks import python

class PersonProcessor:
    @staticmethod
    def detect_people(
        image_np: np.ndarray, max_people: int
    ) -> List[Tuple[int, int, int, int]]:
        base_options = python.BaseOptions(
            model_asset_path="thirdparty/ssd_mobilenet_v2.tflite"
        )
        options = ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.4,
            category_allowlist=["person"],
            max_results=max_people,
        )
        detector = ObjectDetector.create_from_options(options)
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

        return FullBodyProcessor.process_results(
            mapped_pose, left_hand, right_hand, original_size
        )