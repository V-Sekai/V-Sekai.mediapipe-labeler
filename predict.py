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
import sys
import cog
import subprocess
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from cog import BasePredictor, Input, Path, BaseModel, File
import torch
import csv
import os
from PIL import Image

landmark_names = {
    0: "nose",
    1: "left eye (inner)",
    2: "left eye",
    3: "left eye (outer)",
    4: "right eye (inner)",
    5: "right eye",
    6: "right eye (outer)",
    7: "left ear",
    8: "right ear",
    9: "mouth (left)",
    10: "mouth (right)",
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    17: "left pinky",
    18: "right pinky",
    19: "left index",
    20: "right index",
    21: "left thumb",
    22: "right thumb",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
    29: "left heel",
    30: "right heel",
    31: "left foot index",
    32: "right foot index"
}

class Output(BaseModel):
    blendshapes: str
    debug_image: Path
    vrm_skeleton: str = None

class Predictor(BasePredictor):
    def setup(self):
        pass
    
    def draw_skeleton_landmarks_on_image(self, rgb_image, pose_landmarks):
        annotated_image = np.copy(rgb_image)
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
        return annotated_image


    def draw_landmarks_on_image(self, rgb_image, detection_result=None, pose_landmarks=None):
        annotated_image = np.copy(rgb_image)

        # Draw face landmarks if we have a detection result
        if detection_result is not None and detection_result.face_landmarks:
            face_landmarks_list = detection_result.face_landmarks
            for idx in range(len(face_landmarks_list)):
                face_landmarks = face_landmarks_list[idx]
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in face_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
                solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
        # Draw pose landmarks if specified
        if pose_landmarks is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

        return annotated_image


    def run(self, image_path, predict_vrm_skeleton=False):
        base_options = python.BaseOptions(model_asset_path='thirdparty/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        image = mp.Image.create_from_file(str(image_path))
        detection_result = detector.detect(image)

        orig_image = Image.open(image_path).convert('RGB')
        annotated_image = self.draw_landmarks_on_image(np.array(orig_image), detection_result)
        annotated_pil_image = Image.fromarray(annotated_image)
        opencv_image = cv2.cvtColor(np.array(annotated_pil_image), cv2.COLOR_RGB2BGR)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        debug_img_path = os.path.join(script_dir, f"{os.path.basename(image_path)}_debug.jpg")
        cv2.imwrite(debug_img_path, opencv_image)

        blendshapes = []
        if detection_result.face_blendshapes:
            for category in detection_result.face_blendshapes[0]:
                score = '{:.20f}'.format(category.score) if abs(category.score) > 1e-9 else '0.00000000000000000000'
                blendshapes.append({"category_name": category.category_name, "score": score})

        # Predict skeleton if requested
        skeleton_data = []
        if predict_vrm_skeleton:
            with mp.solutions.pose.Pose(static_image_mode=True) as pose:
                skeleton_img = cv2.imread(str(image_path))
                results = pose.process(cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    annotated_skeleton = self.draw_landmarks_on_image(skeleton_img, pose_landmarks=results.pose_landmarks)
                    debug_skeleton_path = os.path.join(
                        script_dir,
                        f"{os.path.basename(image_path)}_skeleton_debug.jpg"
                    )
                    cv2.imwrite(debug_skeleton_path, annotated_skeleton)
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        skeleton_data.append({
                            "name": landmark_names.get(i, f"unknown_{i}"),
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.visibility
                        })

        out = Output(
            debug_image=Path(debug_img_path),
            blendshapes=json.dumps(blendshapes, sort_keys=True),
            vrm_skeleton=json.dumps(skeleton_data) if predict_vrm_skeleton else None
        )
        return out

    def predict(self,
                image_path: Path = Input(description="RGB input image"),
                predict_vrm_skeleton: bool = Input(default=True)) -> Output:
        return self.run(image_path, predict_vrm_skeleton=predict_vrm_skeleton)

if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    result = predictor.predict(image_path)
    print(result)