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
from cog import BasePredictor, Input, Path
import torch
import csv
import os

class Predictor(BasePredictor):
    def setup(self):
        pass

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
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

        return annotated_image

    def predict(self, image_path: Path = Input(description="RGB input image")) -> dict[str, Path]:
        base_options = python.BaseOptions(model_asset_path='thirdparty/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)
        image = mp.Image.create_from_file(str(image_path))
        detection_result = detector.detect(image)
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)

        script_dir = os.path.dirname(os.path.realpath(__file__))

        debug_img_path = os.path.join(script_dir, f"{os.path.basename(image_path)}_debug.jpg")
        cv2.imwrite(debug_img_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        blendshapes = [{"category_name": category.category_name, "score": '{:.20f}'.format(category.score) if abs(category.score) > 1e-9 else '0.00000000000000000000'} for category in detection_result.face_blendshapes[0]]
        blendshapes_json_path = os.path.join(script_dir, f"{os.path.basename(image_path)}_blendshapes.json")

        with open(blendshapes_json_path, 'w') as f:
            json.dump(blendshapes, f, indent=4, sort_keys=True)

        return {
            "blendshapes_json": Path(blendshapes_json_path),
            "debug_img": Path(debug_img_path)
        }

if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    result = predictor.predict(image_path)
    print(result)