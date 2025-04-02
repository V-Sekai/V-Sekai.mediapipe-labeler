import os
import json
import urllib.request
import numpy as np
from datetime import datetime
from PIL import ImageDraw, ImageFont
from cog import BasePredictor, Input, Path
from typing import Any
from filters import OneEuroFilter
from config import COCO_KEYPOINT_NAMES, HAND_MAPPINGS, FACS_CONFIG
from models import Output
from utils.media_utils import MediaUtils
from utils.annotation_utils import AnnotationUtils
from processors.pose_processor import PoseProcessor
from processors.face_processor import FaceProcessor
from processors.hand_processor import HandProcessor
from processors.person_processor import PersonProcessor
from processors.full_body_processor import FullBodyProcessor
from processors.facs_processor import FACSAnalyzer
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
from PIL import Image, ImageDraw

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
                    model_asset_path="thirdparty/face_landmarker.task",
                    delegate=python.BaseOptions.Delegate.CPU,
                ),
                output_face_blendshapes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self.pose_processor = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            smooth_landmarks=True,
        )

        self.hand_processor = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="thirdparty/hand_landmarker.task",
                    delegate=python.BaseOptions.Delegate.CPU,
                ),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        # Initialize 1€ filters for each keypoint coordinate
        self.filters_x = []
        self.filters_y = []
        num_keypoints = len(COCO_KEYPOINT_NAMES)
        mincutoff = 1.0  # Lower values = more smoothing
        beta = 0.7  # Higher values = more responsive to movement
        dcutoff = 1.0
        initial_freq = 30

        for _ in range(num_keypoints):
            self.filters_x.append(OneEuroFilter(initial_freq, mincutoff, beta, dcutoff))
            self.filters_y.append(OneEuroFilter(initial_freq, mincutoff, beta, dcutoff))

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
            description="Process every nth frame for video input",
            ge=1,
            default=1,
        ),
    ) -> Output:
        media_type = "image"
        if str(media_path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpv")):
            media_type = "video"
            return self.process_video(media_path, max_people, frame_sample_rate)
        else:
            return self.process_image(media_path, max_people)

    def process_image(self, image_path: Path, max_people: int) -> Output:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        original_h, original_w = img_np.shape[:2]

        boxes = PersonProcessor.detect_people(img_np, max_people)
        all_results = []

        for person_id, box in enumerate(boxes):
            startX, startY, endX, endY = box
            crop = img_np[startY:endY, startX:endX]

            if crop.size == 0:
                continue

            person_result = PersonProcessor.process_crop(
                crop, box, (original_h, original_w), self
            )

            if person_result:
                person_result["person_id"] = person_id
                person_result["box"] = box
                all_results.append(person_result)

        return Output(
            coco_keypoints=json.dumps(
                self.aggregate_coco(all_results, original_w, original_h), indent=2
            ),
            facs=json.dumps({"people": [r["facs"] for r in all_results]}, indent=2),
            fullbodyfacs=json.dumps(
                {"people": [r["fullbodyfacs"] for r in all_results]}, indent=2
            ),
            debug_media=self.create_debug_image(img_np, all_results),
            hand_landmarks=json.dumps([r["hands"] for r in all_results])
            if any(r["hands"] for r in all_results)
            else None,
            num_people=len(all_results),
            media_type="image",
        )

    def process_video(
        self, video_path: Path, max_people: int, frame_sample_rate: int
    ) -> Output:
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup intermediate video writer (no audio)
        debug_video_path = "/tmp/debug_output_intermediate.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            debug_video_path, fourcc, fps / frame_sample_rate, (width, height)
        )

        frame_results = []
        frame_count = 0
        processed_count = 0
        max_people_detected = 0

        progress = tqdm(
            total=total_frames,
            desc="Processing video",
            unit="frame",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            progress.update(1)

            if frame_count % frame_sample_rate != 0:
                frame_count += 1
                continue

            img_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = PersonProcessor.detect_people(img_np, max_people)
            all_results = []

            for person_id, box in enumerate(boxes):
                startX, startY, endX, endY = box
                crop = img_np[startY:endY, startX:endX]

                if crop.size == 0:
                    continue

                person_result = PersonProcessor.process_crop(
                    crop, box, (height, width), self
                )

                if person_result:
                    person_result["person_id"] = person_id
                    person_result["box"] = box
                    all_results.append(person_result)

            # Update max people count
            current_people = len(all_results)
            if current_people > max_people_detected:
                max_people_detected = current_people

            # Collect frame results
            frame_results.append(
                {
                    "coco": self.aggregate_coco(all_results, width, height),
                    "facs": [r["facs"] for r in all_results],
                    "fullbodyfacs": [r["fullbodyfacs"] for r in all_results],
                    "hands": [r["hands"] for r in all_results],
                    "num_people": current_people,
                }
            )

            # Annotate and write frame
            annotated_frame = self.annotate_video_frame(img_np, all_results)
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            processed_count += 1
            frame_count += 1

        progress.close()
        cap.release()
        out.release()

        # Convert video format
        final_video_path = "/tmp/debug_output_final.mp4"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    debug_video_path,
                    "-i",
                    str(video_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-movflags",
                    "+faststart",
                    "-shortest",
                    final_video_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.remove(debug_video_path)
        except Exception as e:
            print(f"Video conversion failed: {e}")
            final_video_path = debug_video_path

        return Output(
            coco_keypoints=json.dumps([f["coco"] for f in frame_results], indent=2),
            facs=json.dumps(
                {"frames": [{"people": f["facs"]} for f in frame_results]}, indent=2
            ),
            fullbodyfacs=json.dumps(
                {"frames": [{"people": f["fullbodyfacs"]} for f in frame_results]},
                indent=2,
            ),
            debug_media=Path(final_video_path),
            hand_landmarks=json.dumps([f["hands"] for f in frame_results], indent=2),
            num_people=max_people_detected,
            media_type="video",
            total_frames=processed_count,
        )

    def annotate_video_frame(self, frame: np.ndarray, results: list) -> np.ndarray:
        annotated = Image.fromarray(frame)
        draw = ImageDraw.Draw(annotated)
        colors = {
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "red": (0, 0, 255),
            "orange": (255, 165, 0),
            "yellow": (255, 255, 0),
            "magenta": (255, 0, 255),
        }

        for result in results:
            startX, startY, endX, endY = result["box"]
            # Draw bounding box
            draw.rectangle(
                [(startX, startY), (endX, endY)], outline=colors["green"], width=2
            )

            # Draw skeleton and keypoints
            keypoints = result["fullbodyfacs"]["keypoints"]
            self.draw_skeleton(draw, keypoints, colors)

            # Draw person ID label
            label = f"Person {result['person_id']}"
            draw.text((startX, startY - 20), label, fill=colors["green"])

        return np.array(annotated)

    def create_debug_image(self, img_np: np.ndarray, all_results: list) -> Path:
        annotated = Image.fromarray(img_np)
        draw = ImageDraw.Draw(annotated)
        colors = {
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "red": (0, 0, 255),
            "orange": (255, 165, 0),
            "yellow": (255, 255, 0),
            "magenta": (255, 0, 255),
        }

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()

        for result in all_results:
            startX, startY, endX, endY = result["box"]
            draw.rectangle(
                [(startX, startY), (endX, endY)], outline=colors["green"], width=2
            )

            keypoints = result["fullbodyfacs"]["keypoints"]
            self.draw_skeleton(draw, keypoints, colors)

            label = f"Person {result['person_id']}"
            draw.text((startX, startY - 20), label, fill=colors["green"], font=font)

        debug_path = "/tmp/debug_output.jpg"
        annotated.save(debug_path)
        return Path(debug_path)

    def draw_skeleton(self, draw, keypoints, colors):
        kp_dict = {kp["id"]: kp for kp in keypoints}
        connections = []
        for kp in keypoints:
            parent_id = kp.get("parent", -1)
            if parent_id != -1 and parent_id in kp_dict:
                parent = kp_dict[parent_id]
                connections.append((parent, kp))

        for parent, child in connections:
            parent_vis = parent.get("visibility", 1.0)
            child_vis = child.get("visibility", 1.0)
            if parent_vis < 0.5 or child_vis < 0.5:
                continue
            x1, y1 = parent["position"][0], parent["position"][1]
            x2, y2 = child["position"][0], child["position"][1]
            draw.line([(x1, y1), (x2, y2)], fill=colors["orange"], width=2)

        for kp in keypoints:
            x, y = kp["position"][0], kp["position"][1]
            bbox = [(x - 4, y - 4), (x + 4, y + 4)]
            color = colors["blue"] if kp["id"] < 33 else colors["red"]
            draw.ellipse(bbox, fill=color, outline=None)

    def aggregate_coco(self, results, width, height):
        annotations = []
        for idx, res in enumerate(results):
            ann = res["coco"]["annotations"][0].copy()
            ann["id"] = idx
            annotations.append(ann)

        return {
            "info": {
                "description": "Multi-person COCO 1.1 Extended",
                "version": "1.1",
                "year": 2023,
                "contributor": "MediaPipe Crowd Processor",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [{"id": 1, "name": "CC-BY-4.0"}],
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
            "annotations": annotations,
            "categories": FullBodyProcessor.create_coco_output(None, None, (0, 0))[
                "categories"
            ],
        }

if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    result = predictor.predict(Path("input.mp4"))
    print(json.dumps(json.loads(result.json()), indent=2))
