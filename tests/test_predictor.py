import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import zipfile
import json
import tempfile
import numpy as np
from PIL import Image
import cv2
import numpy as np

from src.predict import (
    Predictor,
    PersonProcessor,
)


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor()
        self.predictor.setup = MagicMock()
        self.predictor.face_processor = MagicMock()
        self.predictor.pose_processor = MagicMock()
        self.predictor.hand_processor = MagicMock()

    def create_test_image(self):
        img = Image.new("RGB", (640, 480), color="white")
        tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp_img.name)
        return tmp_img

    def create_test_video(self):
        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_video.name, fourcc, 1, (640, 480))
        for _ in range(5):  # 5 frames
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        return tmp_video

    @patch.object(PersonProcessor, "detect_people")
    @patch.object(PersonProcessor, "process_crop")
    def test_image_processing(self, mock_process_crop, mock_detect_people):
        # Configure mocks
        mock_detect_people.return_value = [(50, 50, 200, 200)]
        mock_process_crop.return_value = {
            "mediapipe": {
                "annotations": [
                    {
                        "keypoints": [0.0] * 135,
                        "num_keypoints": 10,
                        "bbox": [50.0, 50.0, 150.0, 150.0],
                        "area": 22500.0,
                    }
                ],
                "categories": [
                    {
                        "id": 1,
                        "name": "person",
                        "supercategory": "person",
                        "keypoints": [],
                        "skeleton": [],
                    }
                ],
            }
        }

        # Create test image
        with self.create_test_image() as tmp_img:
            output = self.predictor.process_image(
                Path(tmp_img.name), max_people=1, test_mode=False
            )

            # Verify ZIP outputs
            self._verify_result_zip(output.result_zip, expected_images=1)
            self._verify_debug_zip(output.debug_zip)

    @patch.object(PersonProcessor, "detect_people")
    @patch.object(PersonProcessor, "process_crop")
    def test_video_processing(self, mock_process_crop, mock_detect_people):
        # Configure mocks
        mock_detect_people.return_value = [(50, 50, 200, 200)]
        mock_process_crop.return_value = {
            "mediapipe": {
                "annotations": [
                    {
                        "keypoints": [0.0] * 135,
                        "num_keypoints": 10,
                        "bbox": [50.0, 50.0, 150.0, 150.0],
                        "area": 22500.0,
                    }
                ],
                "categories": [
                    {
                        "id": 1,
                        "name": "person",
                        "supercategory": "person",
                        "keypoints": [],
                        "skeleton": [],
                    }
                ],
            }
        }

        # Create test video
        with self.create_test_video() as tmp_video:
            output = self.predictor.process_video(
                Path(tmp_video.name), max_people=1, frame_sample_rate=1, test_mode=False
            )

            # Verify ZIP outputs
            self._verify_result_zip(output.result_zip, expected_images=5)
            self._verify_debug_zip(output.debug_zip, is_video=True)
            self.assertEqual(output.total_frames, 5)

    def _verify_result_zip(self, zip_path, expected_images):
        with zipfile.ZipFile(zip_path, "r") as zipf:
            # Check that there is at least one file in the "train/" folder
            namelist = zipf.namelist()
            self.assertTrue(
                any(n.startswith("train/") for n in namelist),
                "No files found in train/ directory"
            )
            self.assertIn("annotations/annotations.json", namelist)

            # Check image files
            png_files = [
                n for n in namelist if n.startswith("train/") and n.endswith(".png")
            ]
            self.assertEqual(len(png_files), expected_images)

            # Check annotations
            with zipf.open("annotations/annotations.json") as f:
                annotations = json.load(f)
                self.assertIn("images", annotations)
                self.assertIn("annotations", annotations)
                self.assertEqual(len(annotations["images"]), expected_images)
                self.assertGreaterEqual(
                    len(annotations["annotations"]), expected_images
                )

    def _verify_debug_zip(self, zip_path, is_video=False):
        with zipfile.ZipFile(zip_path, "r") as zipf:
            namelist = zipf.namelist()
            if is_video:
                self.assertTrue(any(n.endswith(".mp4") for n in namelist))
            else:
                self.assertTrue(any(n.endswith(".jpg") for n in namelist))

    @patch.object(PersonProcessor, "detect_people")
    def test_no_people_detected(self, mock_detect_people):
        mock_detect_people.return_value = []

        with self.create_test_image() as tmp_img:
            output = self.predictor.process_image(
                Path(tmp_img.name), max_people=1, test_mode=False
            )

            with zipfile.ZipFile(output.result_zip, "r") as zipf:
                with zipf.open("annotations/annotations.json") as f:
                    annotations = json.load(f)
                    self.assertEqual(len(annotations["annotations"]), 0)

    @patch.object(PersonProcessor, "detect_people")
    def test_test_mode(self, mock_detect_people):
        mock_detect_people.return_value = [(50, 50, 200, 200)] * 3  # 3 people

        with self.create_test_image() as tmp_img:
            output = self.predictor.process_image(
                Path(tmp_img.name), max_people=100, test_mode=True
            )

            with zipfile.ZipFile(output.result_zip, "r") as zipf:
                with zipf.open("annotations/annotations.json") as f:
                    annotations = json.load(f)
                    self.assertEqual(
                        len(annotations["annotations"]), 1
                    )  # Test mode limits to 1


if __name__ == "__main__":
    unittest.main()
