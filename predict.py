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

import tempfile
import cv2
from tqdm import tqdm
import copy
import json
from PIL import ImageDraw
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cog import BasePredictor, Input, Path
from PIL import Image
import shutil
from filters import OneEuroFilter
from models import Output
from full_body_processor import (
    MEDIAPIPE_KEYPOINT_NAMES)
from person_processor import PersonProcessor
from full_body_processor import FullBodyProcessor

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
                min_face_detection_confidence=0.7,
            )
        )
        dummy_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.zeros((100, 100, 3), dtype=np.uint8),
        )
        face_result = self.face_processor.detect(dummy_image)
        self.blendshape_names = (
            [bs.category_name for bs in face_result.face_blendshapes[0]]
            if face_result.face_blendshapes
            else []
        )
        self.pose_processor = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.1,
        )
        self.hand_processor = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="thirdparty/hand_landmarker.task"
                ),
                num_hands=2,
                min_hand_detection_confidence=0.7,
            )
        )
        self.initialize_filters()

    def initialize_filters(self):
        num_keypoints = len(MEDIAPIPE_KEYPOINT_NAMES)
        self.keypoint_filters = [
            {
                "x": OneEuroFilter(10, 1.0, 0.7, 1.0),
                "y": OneEuroFilter(10, 1.0, 0.7, 1.0),
                "z": OneEuroFilter(10, 1.0, 0.7, 1.0),
                "vis": OneEuroFilter(10, 1.0, 0.7, 1.0),
            }
            for _ in range(num_keypoints)
        ]
        self.blendshape_filters = {
            name: OneEuroFilter(30, 1.0, 0.7, 1.0) for name in self.blendshape_names
        }
        self.hand_filters = {
            "left": [
                {
                    "x": OneEuroFilter(10, 1.0, 0.7, 1.0),
                    "y": OneEuroFilter(10, 1.0, 0.7, 1.0),
                    "z": OneEuroFilter(10, 1.0, 0.7, 1.0),
                }
                for _ in range(21)
            ],
            "right": [
                {
                    "x": OneEuroFilter(10, 1.0, 0.7, 1.0),
                    "y": OneEuroFilter(10, 1.0, 0.7, 1.0),
                    "z": OneEuroFilter(10, 1.0, 0.7, 1.0),
                }
                for _ in range(21)
            ],
        }

    def predict(
        self,
        media_path: Path = Input(description="Input image or video file"),
        max_people: int = Input(
            description="Maximum number of people to detect (1-100)",
            ge=1,
            le=100,
            default=100,
        ),
        frame_sample_rate: int = Input(
            description="Process every nth frame for video input", ge=1, default=1
        ),
        test_mode: bool = Input(
            description="Enable test mode for quick verification", default=False
        ),
        export_train: bool = Input(
            description="Export training zip containing json annotations and frame pngs", default=True
        ),
        aligned_media: Path = Input(
            description="Optional video that is aligned with the input video's annotations", 
            default=None
        )
    ) -> Output:
        if str(media_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return self.process_video(
                media_path, max_people, frame_sample_rate, test_mode, export_train, aligned_media
            )
        else:
            return self.process_image(media_path, max_people, test_mode, export_train)

    def process_image(
        self, image_path: Path, max_people: int, test_mode: bool, export_train: bool
    ) -> Output:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        original_h, original_w = img_np.shape[:2]
        boxes = PersonProcessor.detect_people(img_np, max_people)
        if test_mode:
            boxes = boxes[:1]
        all_results = []
        for person_id, box in enumerate(boxes[:max_people]):
            startX, startY, endX, endY = box
            crop = img_np[startY:endY, startX:endX]
            result = PersonProcessor.process_crop(
                crop, box, (original_h, original_w), self
            )
            if result:
                result.update(
                    {
                        "person_id": person_id,
                        "box": (int(startX), int(startY), int(endX), int(endY)),
                    }
                )
                all_results.append(result)
        json_data = {
            "image": {"width": original_w, "height": original_h},
            "annotations": [
                {
                    "bbox": r["mediapipe"]["annotations"][0]["bbox"],
                    "keypoints": r["mediapipe"]["annotations"][0]["keypoints"],
                    "box": r["box"],
                }
                for r in all_results
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            Image.fromarray(img_np).save(tmp_img.name)
            train_img = Path(tmp_img.name)
        debug_media = train_img
        export_folder = None
        if export_train:
            export_folder = self.export_train_folder(json_data, [train_img])
        return Output(
            annotations=json.dumps(json_data),
            debug_media=debug_media,
            num_people=len(all_results),
            media_type="image",
            export_train_folder=export_folder
        )

    def process_video(
        self, video_path: Path, max_people: int, frame_sample_rate: int, test_mode: bool, export_train: bool, aligned_media: Path = None
    ) -> Output:
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        json_data = {
            "metadata": {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_sample_rate": frame_sample_rate,
            },
            "frames": [],
        }
        debug_video_path = "annotated_video.mp4"
        debug_writer = cv2.VideoWriter(
            debug_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if aligned_media is not None:
            color_cap = cv2.VideoCapture(str(aligned_media))
        else:
            color_cap = None
        train_frames = []
        debug_frames = []  # new list to store annotated frames
        frame_count = 0
        processed_count = 0
        with tqdm(total=total_frames, desc="Processing Video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Read corresponding color frame if available
                if color_cap is not None:
                    ret_color, frame_color = color_cap.read()
                    if not ret_color:
                        frame_color = frame
                else:
                    frame_color = frame
                if frame_count % frame_sample_rate == 0:
                    img_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes = PersonProcessor.detect_people(img_np, max_people)
                    frame_raw = []
                    frame_filtered = []
                    for person_id, box in enumerate(boxes):
                        result = PersonProcessor.process_crop(
                            img_np[box[1] : box[3], box[0] : box[2]],
                            box,
                            (height, width),
                            self,
                        )
                        if result:
                            frame_raw.append({
                                "box": box,
                                "mediapipe": result["mediapipe"],
                                "fullbody": result["fullbody"],
                            })
                            filtered = copy.deepcopy(result)
                            self.apply_filters(filtered, frame_count / fps)
                            filtered["person_id"] = person_id
                            filtered["box"] = box
                            frame_filtered.append(filtered)
                    json_data["frames"].append({
                        "frame_number": frame_count,
                        "timestamp": frame_count / fps,
                        "annotations": [
                            {
                                "bbox": r["mediapipe"]["annotations"][0]["bbox"],
                                "keypoints": r["mediapipe"]["annotations"][0]["keypoints"],
                                "fullbody": r["fullbody"],
                            }
                            for r in frame_raw
                        ],
                    })
                    frame_color_rgb = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB)
                    annotated_frame = self.annotate_video_frame(frame_color_rgb, frame_filtered)
                    debug_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                    debug_frames.append(annotated_frame)  # store the annotated frame
                    if export_train:
                        frame_png = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
                        Image.fromarray(frame_color_rgb).save(frame_png)
                        train_frames.append(frame_png)
                    processed_count += 1
                    pbar.update(1)
                frame_count += 1
                if test_mode and frame_count >= 5 * frame_sample_rate:
                    break
        cap.release()
        debug_writer.release()
        if color_cap is not None:
            color_cap.release()
        new_video = "debug_video.mp4"
        writer = cv2.VideoWriter(new_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frm in debug_frames:
            writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
        writer.release()
        debug_media = Path(new_video)
        annotations_data = json_data
        export_folder = None
        if export_train:
            export_folder = self.export_train_folder(json_data, train_frames)
        return Output(
            annotations=json.dumps(annotations_data),
            debug_media=debug_media,
            num_people=max((len(f["annotations"]) for f in json_data["frames"]), default=0),
            media_type="video",
            total_frames=processed_count,
            export_train_folder=export_folder
        )

    def export_train_folder(self, json_data, frame_files: list) -> Path:
        from full_body_processor import MEDIAPIPE_KEYPOINT_NAMES, FullBodyProcessor
        from datetime import datetime
        current_time = datetime.now().isoformat()
        def convert_to_coco(json_data, frame_files):
            coco = {
                "info": {
                    "year": datetime.now().year,
                    "version": "4",
                    "description": "Exported from roboflow.com",
                    "contributor": "K. S. Ernest (iFire) Lee <ernest.lee@chibifire.com>",
                    "url": "https://app.roboflow.com/datasets/person-keyposition-ecazz",
                    "date_created": current_time
                },
                "licenses": [
                    {"id": 1, "url": "", "name": "Unknown"}
                ],
                "categories": [],
                "images": [],
                "annotations": []
            }
            category = {
                "id": 1,
                "name": "person-full-body-facs",
                "supercategory": "persons",
                "keypoints": MEDIAPIPE_KEYPOINT_NAMES,
                "skeleton": FullBodyProcessor.SKELETON_CONNECTIONS,
            }
            coco["categories"].append(category)
            ann_id = 1
            image_id = 0
            if "image" in json_data:
                image_id = 0
                file_name = os.path.basename(frame_files[0])
                image_info = {
                    "id": image_id,
                    "license": 1,
                    "width": json_data["image"]["width"],
                    "height": json_data["image"]["height"],
                    "file_name": file_name,
                    "date_captured": current_time,
                    "extra": {}
                }
                coco["images"].append(image_info)
                for ann in json_data["annotations"]:
                    coco_ann = {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": ann["bbox"],
                        "area": ann["bbox"][2] * ann["bbox"][3],
                        "iscrowd": 0,
                        "keypoints": ann["keypoints"],
                        "num_keypoints": len([k for k in ann["keypoints"] if k != 0])
                    }
                    coco["annotations"].append(coco_ann)
                    ann_id += 1
            else:
                for frame_ann in json_data["frames"]:
                    file_name = "frame_{:06d}.png".format(frame_ann["frame_number"])
                    image_info = {
                        "id": image_id,
                        "license": 1,
                        "width": json_data["metadata"]["width"],
                        "height": json_data["metadata"]["height"],
                        "file_name": file_name,
                        "date_captured": current_time,
                        "extra": {}
                    }
                    coco["images"].append(image_info)
                    for ann in frame_ann["annotations"]:
                        coco_ann = {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": ann["bbox"],
                            "area": ann["bbox"][2] * ann["bbox"][3],
                            "iscrowd": 0,
                            "keypoints": ann["keypoints"],
                            "num_keypoints": len([k for k in ann["keypoints"] if k != 0])
                        }
                        coco["annotations"].append(coco_ann)
                        ann_id += 1
                    image_id += 1
            return coco

        coco_data = convert_to_coco(json_data, frame_files)
        # Create temporary directory and write COCO JSON inside train folder
        with tempfile.TemporaryDirectory(prefix="export_train_") as temp_dir:
            train_dir = os.path.join(temp_dir, "train")
            os.makedirs(train_dir, exist_ok=True)
            # Copy all frame files into the train folder
            for f in frame_files:
                shutil.copy(f, os.path.join(train_dir, os.path.basename(f)))
            # Write the COCO JSON to _annotations.coco.json inside train folder
            json_path = os.path.join(train_dir, "_annotations.coco.json")
            with open(json_path, "w") as f:
                json.dump(coco_data, f, indent=2)
            # Create a NamedTemporaryFile for the zip archive
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                zip_base = tmp_zip.name.replace(".zip", "")
            zip_path = shutil.make_archive(base_name=zip_base, format='zip', root_dir=temp_dir)
            return Path(zip_path)

    def apply_filters(self, person_data, timestamp):
        for kp in person_data["fullbody"]["keypoints"]:
            kp_id = kp["id"]
            if kp_id >= len(self.keypoint_filters):
                continue
            x = kp["position"][0]
            y = kp["position"][1]
            z = kp["position"][2]
            vis = kp["visibility"]
            kp["position"] = [
                self.keypoint_filters[kp_id]["x"](x, timestamp),
                self.keypoint_filters[kp_id]["y"](y, timestamp),
                self.keypoint_filters[kp_id]["z"](z, timestamp),
            ]
            kp["visibility"] = self.keypoint_filters[kp_id]["vis"](vis, timestamp)
        filtered_blendshapes = []
        for bs in person_data["blendshapes"]:
            name = bs["name"]
            if name in self.blendshape_filters:
                filtered_score = self.blendshape_filters[name](bs["score"], timestamp)
                filtered_blendshapes.append({"name": name, "score": filtered_score})
        person_data["blendshapes"] = filtered_blendshapes
        for hand_type in ["left", "right"]:
            hand = person_data["hands"].get(hand_type, [])
            for idx, landmark in enumerate(hand):
                if idx >= 21:
                    continue
                filters = self.hand_filters[hand_type][idx]
                x = landmark.get("x", 0.0)
                y = landmark.get("y", 0.0)
                z = landmark.get("z", 0.0)
                landmark["x"] = filters["x"](x, timestamp)
                landmark["y"] = filters["y"](y, timestamp)
                landmark["z"] = filters["z"](z, timestamp)

    def annotate_video_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
        annotated = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated)
        colors = {"green": (0, 255, 0), "red": (0, 0, 255), "orange": (255, 165, 0)}
        for result in results:
            box = result.get("box", (0, 0, 0, 0))
            startX, startY, endX, endY = map(int, box)
            draw.rectangle([(startX, startY), (endX, endY)], outline=colors["green"], width=2)
            if "fullbody" in result:
                keypoints = result["fullbody"]["keypoints"]
                self.draw_skeleton(draw, keypoints, colors)
        return np.array(annotated)

    def draw_skeleton(self, draw, keypoints, colors):
        kp_dict = {kp["id"]: kp for kp in keypoints}
        for connection in FullBodyProcessor.SKELETON_CONNECTIONS:
            if connection[0] in kp_dict and connection[1] in kp_dict:
                parent = kp_dict[connection[0]]
                child = kp_dict[connection[1]]
                if parent["visibility"] < 0.5 or child["visibility"] < 0.5:
                    continue
                x1, y1 = parent["position"][0], parent["position"][1]
                x2, y2 = child["position"][0], child["position"][1]
                draw.line([(x1, y1), (x2, y2)], fill=colors["orange"], width=2)
        for kp in keypoints:
            if kp["visibility"] < 0.5 or kp["id"] > 32:
                continue
            x, y = kp["position"][0], kp["position"][1]
            bbox = [(x - 4, y - 4), (x + 4, y + 4)]
            draw.ellipse(bbox, fill=colors["red"], outline=None)

    def aggregate_mediapipe(self, results, width, height):
        annotations = []
        for res in results:
            ann = res["mediapipe"]["annotations"][0].copy()
            annotations.append(
                {
                    "keypoints": ann["keypoints"],
                    "num_keypoints": ann["num_keypoints"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "category_id": 1,
                    "iscrowd": 0,
                }
            )
        return {"annotations": annotations}
