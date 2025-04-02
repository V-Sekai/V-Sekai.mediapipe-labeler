#!/usr/bin/env python3
import cv2
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from cog import BasePredictor, Path, BaseModel
import mediapipe as mp

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_PARENT = {
    0: -1,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 0,
    6: 0,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
    11: 5,
    12: 6,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
}


class Output(BaseModel):
    coco_keypoints: str
    blendshapes: str
    debug_media: Path
    num_people: int
    media_type: str
    total_frames: Optional[int] = None


class Predictor(BasePredictor):
    def setup(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.skeleton_connections = [(i, p) for i, p in COCO_PARENT.items() if p != -1]

    def predict(self, media_path: Path, frame_sample: int = 1) -> Output:
        is_video = media_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".mpv")
        return (
            self.process_video(media_path, frame_sample)
            if is_video
            else self.process_image(media_path)
        )

    def process_image(self, image_path: Path) -> Output:
        image = cv2.imread(str(image_path))
        results = self.detect_poses(image)
        self._save_debug(image, results, "/tmp/debug.jpg")
        return Output(
            coco_keypoints=json.dumps(self._format_coco(results, *image.shape[:2])),
            blendshapes=json.dumps({"people": []}),
            debug_media=Path("/tmp/debug.jpg"),
            num_people=len(results),
            media_type="image",
        )

    def process_video(self, video_path: Path, frame_sample: int) -> Output:
        cap = cv2.VideoCapture(str(video_path))
        width, height = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(
            "/tmp/debug.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(5) / frame_sample,
            (width, height),
        )

        frame_results = []
        for idx in range(int(cap.get(7))):
            if idx % frame_sample == 0 and cap.grab():
                _, frame = cap.retrieve()
                results = self.detect_poses(frame)
                self._save_debug(frame, results, None, writer)
                frame_results.append(results)

        cap.release()
        writer.release()
        return Output(
            coco_keypoints=json.dumps(
                [self._format_coco(r, width, height) for r in frame_results]
            ),
            blendshapes=json.dumps({"frames": []}),
            debug_media=Path("/tmp/debug.mp4"),
            num_people=max(len(r) for r in frame_results),
            media_type="video",
            total_frames=len(frame_results),
        )

    def detect_poses(self, image: np.ndarray) -> List[Dict]:
        pose_data = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return (
            [self._process_landmarks(pose_data.pose_landmarks, image.shape)]
            if pose_data.pose_landmarks
            else []
        )

    def _process_landmarks(self, landmarks, shape):
        return {
            "keypoints": [
                coord
                for lmk in landmarks.landmark
                for coord in (
                    lmk.x * shape[1],
                    lmk.y * shape[0],
                    2 * (lmk.visibility > 0.5),
                )
            ]
        }

    def _save_debug(self, frame, results, img_path=None, video_writer=None):
        annotated = frame.copy()
        for result in results:
            kps = result["keypoints"]
            for i, p in self.skeleton_connections:
                pt1 = (int(kps[i * 3]), int(kps[i * 3 + 1]))
                pt2 = (int(kps[p * 3]), int(kps[p * 3 + 1]))
                cv2.line(annotated, pt1, pt2, (0, 165, 255), 2)
            for x, y, _ in zip(kps[::3], kps[1::3], kps[2::3]):
                cv2.circle(annotated, (int(x), int(y)), 4, (255, 0, 0), -1)

        img_path and cv2.imwrite(img_path, annotated)
        video_writer and video_writer.write(annotated)

    def _format_coco(self, results, width, height):
        return {
            "info": {"description": "COCO Pose Results", "version": "1.0"},
            "licenses": [{"id": 1, "name": "CC-BY-4.0"}],
            "images": [self._coco_image(0, width, height)],
            "annotations": [self._coco_annotation(i, r) for i, r in enumerate(results)],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": COCO_KEYPOINT_NAMES,
                    "skeleton": list(COCO_PARENT.items()),
                }
            ],
        }

    def _coco_image(self, img_id, width, height):
        return {
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": "input",
            "date_captured": datetime.now().isoformat(),
        }

    def _coco_annotation(self, ann_id, result):
        return {
            "id": ann_id,
            "image_id": 0,
            "category_id": 1,
            "iscrowd": 0,
            "keypoints": result["keypoints"],
            "num_keypoints": sum(v > 0 for v in result["keypoints"][2::3]),
        }
