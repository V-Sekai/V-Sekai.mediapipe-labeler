from cog import BasePredictor, Input, Path
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
import cv2
import os
import tempfile
import json
from tqdm import tqdm
from .filters import OneEuroFilter
from .processors import PersonProcessor, FullBodyProcessor, MEDIAPIPE_KEYPOINT_NAMES


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

        # Initialize face processor and get blendshape names
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

        # Initialize pose and hand processors
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

        # Initialize filters
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
    ) -> Output:
        if str(media_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return self.process_video(
                media_path, max_people, frame_sample_rate, test_mode
            )
        else:
            return self.process_image(media_path, max_people, test_mode)

    def process_image(
        self, image_path: Path, max_people: int, test_mode: bool
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
                # Add box coordinates to the result
                result.update(
                    {
                        "person_id": person_id,
                        "box": (int(startX), int(startY), int(endX), int(endY)),
                    }
                )
                all_results.append(result)

        # Create annotated debug media
        annotated_frame = self.annotate_video_frame(img_np, all_results)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp_json:
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
            json.dump(json_data, tmp_json)
            annotations_path = Path(tmp_json.name)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
            Image.fromarray(annotated_frame).save(tmp_img.name)
            debug_media = Path(tmp_img.name)

        return Output(
            annotations=annotations_path,
            debug_media=debug_media,
            num_people=len(all_results),
            media_type="image",
        )

    def process_video(
        self, video_path: Path, max_people: int, frame_sample_rate: int, test_mode: bool
    ) -> Output:
        # Video setup
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output setup
        json_data = {
            "metadata": {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_sample_rate": frame_sample_rate,
            },
            "frames": [],
        }

        # Debug video writer
        debug_video_path = "annotated_video.mp4"
        debug_writer = cv2.VideoWriter(
            debug_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        frame_count = 0
        processed_count = 0

        with tqdm(total=total_frames, desc="Processing Video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_sample_rate == 0:
                    img_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes = PersonProcessor.detect_people(img_np, max_people)

                    frame_raw = []
                    frame_filtered = []

                    for person_id, box in enumerate(boxes):
                        # Process raw data
                        result = PersonProcessor.process_crop(
                            img_np[box[1] : box[3], box[0] : box[2]],
                            box,
                            (height, width),
                            self,
                        )
                        if result:
                            # Store raw frame data
                            frame_raw.append(
                                {
                                    "box": box,
                                    "mediapipe": result["mediapipe"],
                                    "fullbody": result["fullbody"],
                                }
                            )

                            # Create filtered copy
                            filtered = copy.deepcopy(result)
                            self.apply_filters(filtered, frame_count / fps)
                            filtered["person_id"] = person_id
                            filtered["box"] = box
                            frame_filtered.append(filtered)

                    # Store raw data
                    json_data["frames"].append(
                        {
                            "frame_number": frame_count,
                            "timestamp": frame_count / fps,
                            "annotations": [
                                {
                                    "bbox": r["mediapipe"]["annotations"][0]["bbox"],
                                    "keypoints": r["mediapipe"]["annotations"][0][
                                        "keypoints"
                                    ],
                                    "fullbody": r["fullbody"],
                                }
                                for r in frame_raw
                            ],
                        }
                    )

                    # Generate and write annotated frame
                    annotated_frame = self.annotate_video_frame(img_np, frame_filtered)
                    debug_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                    processed_count += 1
                    pbar.update(1)

                frame_count += 1
                if test_mode and frame_count >= 5 * frame_sample_rate:
                    break

        # Cleanup
        cap.release()
        debug_writer.release()

        # Save JSON
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp_json:
            json.dump(json_data, tmp_json, indent=2)
            json_path = Path(tmp_json.name)

        return Output(
            annotations=json_path,
            debug_media=Path(debug_video_path),
            num_people=max((len(f["annotations"]) for f in json_data["frames"]), default=0),
            media_type="video",
            total_frames=processed_count,
        )

    def apply_filters(self, person_data, timestamp):
        # Filter keypoints
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

        # Filter blendshapes
        filtered_blendshapes = []
        for bs in person_data["blendshapes"]:
            name = bs["name"]
            if name in self.blendshape_filters:
                filtered_score = self.blendshape_filters[name](bs["score"], timestamp)
                filtered_blendshapes.append({"name": name, "score": filtered_score})
        person_data["blendshapes"] = filtered_blendshapes

        # Filter hands
        for hand_type in ["left", "right"]:
            hand = person_data["hands"].get(hand_type, [])
            for idx, landmark in enumerate(hand):
                if idx >= 21:
                    continue
                filters = self.hand_filters[hand_type][idx]

                x = landmark.get("x", 0.0)
                y = landmark.get("y", 0.0)
                z = landmark.get("z", 0.0)

                # Apply filters
                landmark["x"] = filters["x"](x, timestamp)
                landmark["y"] = filters["y"](y, timestamp)
                landmark["z"] = filters["z"](z, timestamp)

    def annotate_video_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
        annotated = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated)
        colors = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "orange": (255, 165, 0),
        }

        for result in results:
            # Safely get box coordinates with fallback
            box = result.get("box", (0, 0, 0, 0))
            startX, startY, endX, endY = map(int, box)

            # Draw bounding box
            draw.rectangle(
                [(startX, startY), (endX, endY)], outline=colors["green"], width=2
            )

            # Draw skeleton if available
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