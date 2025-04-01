import cv2
import json
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from cog import BasePredictor, Input, Path, BaseModel
from PIL import Image
from typing import Any, Dict, List, Tuple
from datetime import datetime
from collections import deque

# Configuration
COCO_KEYPOINT_NAMES = [
    # 17 Standard COCO Body Keypoints
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
    # 11 Extended Facial Keypoints
    "brow_inner_left",
    "brow_inner_right",
    "brow_outer_left",
    "brow_outer_right",
    "lid_upper_left",
    "lid_upper_right",
    "lid_lower_left",
    "lid_lower_right",
    "lip_upper",
    "lip_lower",
    "lip_corner_left",
    "lip_corner_right",
]
COCO_SKELETON = [
    # Standard COCO Body Connections
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
    # Extended Facial Connections
    [0, 17],
    [0, 18],  # Nose to inner brows
    [17, 19],
    [18, 20],  # Inner to outer brows
    [17, 21],
    [18, 22],  # Brows to upper lids
    [21, 23],
    [22, 24],  # Upper to lower lids
    [0, 25],
    [0, 26],  # Nose to lips
    [25, 27],
    [26, 28],  # Lip corners
]
FACS_AU_MAPPING = {
    "AU1": ["browDownLeft", "browDownRight"],
    "AU2": ["browOuterUpLeft", "browOuterUpRight"],
    "AU4": ["browInnerUp", "browInnerDown"],
    "AU5": ["eyeBlinkLeft", "eyeBlinkRight"],
    "AU6": ["eyeSquintLeft", "eyeSquintRight"],
    "AU9": ["noseSneerLeft", "noseSneerRight"],
    "AU12": ["mouthLeft", "mouthRight"],
    "AU25": ["jawOpen", "mouthStretchLeft"],
}


class Output(BaseModel):
    coco_keypoints: str
    facs: str
    fullbodyfacs: str
    vrm_skeleton: str
    debug_image: Path


class FullBodyProcessor:
    @staticmethod
    def process_results(pose_landmarks, face_landmarks, face_blendshapes, image_size):
        """Process MediaPipe results into all output formats"""
        return {
            "coco": FullBodyProcessor._create_coco_output(
                pose_landmarks, face_landmarks, image_size
            ),
            "facs": FullBodyProcessor._create_facs_output(face_blendshapes),
            "fullbodyfacs": FullBodyProcessor._create_fullbodyfacs(
                pose_landmarks, face_landmarks, image_size
            ),
            "vrm": FullBodyProcessor._create_vrm_skeleton(pose_landmarks),
        }

    @staticmethod
    def _create_coco_output(pose, face, image_size):
        """Generate COCO 1.1 compliant JSON output"""
        height, width = image_size
        keypoints = []
        num_visible = 0
        # Process body keypoints (0-16)
        for idx in range(17):
            if idx < len(pose.landmark):
                lmk = pose.landmark[idx]
                x, y, vis = lmk.x * width, lmk.y * height, lmk.visibility
                keypoints += [x, y, 2 if vis > 0.5 else 1]
                num_visible += 1 if vis > 0 else 0
            else:
                keypoints += [0.0, 0.0, 0]
        # Process facial keypoints (17-28)
        facial_indices = [151, 334, 46, 276, 159, 386, 145, 374, 13, 14, 61, 291]
        for idx in facial_indices:
            if face and idx < len(face):
                lmk = face[idx]
                keypoints += [lmk.x * width, lmk.y * height, 2]
                num_visible += 1
            else:
                keypoints += [0.0, 0.0, 0]
        return {
            "info": {
                "description": "COCO 1.1 Extended with Facial Keypoints",
                "version": "1.1",
                "year": 2023,
                "contributor": "MediaPipe FACS Extension",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "CC-BY-4.0",
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                }
            ],
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
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": num_visible,
                    "bbox": FullBodyProcessor._calculate_bbox(keypoints),
                    "area": width * height,
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": COCO_KEYPOINT_NAMES,
                    "skeleton": COCO_SKELETON,
                }
            ],
        }

    @staticmethod
    def _create_facs_output(blendshapes):
        """Generate FACS-compliant AU intensities"""
        au_scores = {}
        for au, components in FACS_AU_MAPPING.items():
            scores = [bs.score for bs in blendshapes if bs.category_name in components]
            au_scores[au] = sum(scores) / len(scores) if scores else 0.0
        return {
            "AUs": au_scores,
            "blendshapes": [
                {"name": bs.category_name, "score": float(bs.score)}
                for bs in blendshapes
            ],
        }

    @staticmethod
    def _create_fullbodyfacs(pose, face, image_size):
        """Generate FullBodyFACS hierarchy"""
        keypoints = []
        height, width = image_size
        # Body keypoints
        for idx in range(33):
            if idx < len(pose.landmark):
                lmk = pose.landmark[idx]
                keypoints.append(
                    {
                        "id": idx,
                        "name": f"body_{idx}",
                        "position": [lmk.x * width, lmk.y * height, lmk.z * width],
                        "parent": FullBodyProcessor._get_parent(idx),
                    }
                )
        # Facial keypoints
        facial_map = {
            151: 17,
            334: 18,
            46: 19,
            276: 20,
            159: 21,
            386: 22,
            145: 23,
            374: 24,
            13: 25,
            14: 26,
            61: 27,
            291: 28,
        }
        for mp_idx, facs_idx in facial_map.items():
            if face and mp_idx < len(face):
                lmk = face[mp_idx]
                keypoints.append(
                    {
                        "id": facs_idx,
                        "name": COCO_KEYPOINT_NAMES[facs_idx],
                        "position": [lmk.x * width, lmk.y * height, lmk.z * width],
                        "parent": 0,  # All facial points parented to nose
                    }
                )
        return {"keypoints": keypoints}

    @staticmethod
    def _get_parent(idx):
        """Get parent joint index for body keypoints"""
        hierarchy = {
            0: -1,  # Nose
            11: 5,  # Left shoulder
            12: 6,  # Right shoulder
            13: 11,  # Left elbow
            14: 12,  # Right elbow
            15: 13,  # Left wrist
            16: 14,  # Right wrist
            23: 11,  # Left hip
            24: 12,  # Right hip
            25: 23,  # Left knee
            26: 24,  # Right knee
            27: 25,  # Left ankle
            28: 26,  # Right ankle
        }
        return hierarchy.get(idx, -1)

    @staticmethod
    def _create_vrm_skeleton(pose_landmarks):
        """Generate VRM 1.0 compliant skeleton structure"""
        if not pose_landmarks:
            return {"bones": [], "nodes": [], "specVersion": "1.0", "humanoid": True}
        VRM_BONE_MAP = {
            "hips": (23, 24),  # Average of left and right hips
            "spine": (11, 12),  # Midpoint between shoulders
            "chest": (11, 12),  # Same as spine for upper body
            "upperChest": (11, 12),  # Shoulder midpoint
            "neck": 0,  # Nose (approximation)
            "head": 0,  # Nose
            "leftUpperLeg": 23,
            "leftLowerLeg": 25,
            "leftFoot": 27,
            "leftToes": 31,
            "rightUpperLeg": 24,
            "rightLowerLeg": 26,
            "rightFoot": 28,
            "rightToes": 32,
            "leftShoulder": 11,
            "leftUpperArm": 13,
            "leftLowerArm": 15,
            "leftHand": 17,
            "rightShoulder": 12,
            "rightUpperArm": 14,
            "rightLowerArm": 16,
            "rightHand": 18,
        }
        VRM_HIERARCHY = {
            "hips": ["spine", "leftUpperLeg", "rightUpperLeg"],
            "spine": ["chest"],
            "chest": ["upperChest", "leftShoulder", "rightShoulder"],
            "upperChest": ["neck"],
            "neck": ["head"],
            "leftUpperLeg": ["leftLowerLeg"],
            "leftLowerLeg": ["leftFoot"],
            "leftFoot": ["leftToes"],
            "rightUpperLeg": ["rightLowerLeg"],
            "rightLowerLeg": ["rightFoot"],
            "rightFoot": ["rightToes"],
            "leftShoulder": ["leftUpperArm"],
            "leftUpperArm": ["leftLowerArm"],
            "leftLowerArm": ["leftHand"],
            "rightShoulder": ["rightUpperArm"],
            "rightUpperArm": ["rightLowerArm"],
            "rightLowerArm": ["rightHand"],
        }

        def get_position(bone_name):
            indices = VRM_BONE_MAP.get(bone_name)
            if not indices:
                return None

            if isinstance(indices, tuple):
                points = []
                for idx in indices:
                    if idx < len(pose_landmarks.landmark):
                        lmk = pose_landmarks.landmark[idx]
                        points.append([lmk.x, lmk.y, lmk.z])
                return np.mean(points, axis=0).tolist() if points else None
            else:
                if indices >= len(pose_landmarks.landmark):
                    return None
                lmk = pose_landmarks.landmark[indices]
                return [lmk.x, lmk.y, lmk.z]

        bones = []
        nodes = []
        visited = set()
        queue = deque([("hips", None)])
        while queue:
            bone_name, parent = queue.popleft()
            if bone_name in visited:
                continue

            position = get_position(bone_name)
            if not position:
                continue
            # Create bone entry
            bone = {"name": bone_name, "head": position, "tail": None, "parent": parent}

            # Calculate tail from children
            children = VRM_HIERARCHY.get(bone_name, [])
            child_positions = []
            for child in children:
                child_pos = get_position(child)
                if child_pos:
                    child_positions.append(child_pos)
                    queue.append((child, bone_name))

            if child_positions:
                bone["tail"] = np.mean(child_positions, axis=0).tolist()

            bones.append(bone)
            nodes.append({"name": bone_name, "translation": bone["head"]})
            visited.add(bone_name)
        return {"bones": bones, "nodes": nodes, "specVersion": "1.0", "humanoid": True}

    @staticmethod
    def _calculate_bbox(keypoints):
        valid = [
            (keypoints[i], keypoints[i + 1])
            for i in range(0, len(keypoints), 3)
            if keypoints[i + 2] > 0
        ]
        if not valid:
            return [0.0, 0.0, 0.0, 0.0]
        x = [p[0] for p in valid]
        y = [p[1] for p in valid]
        return [
            float(min(x)),
            float(min(y)),
            float(max(x) - min(x)),
            float(max(y) - min(y)),
        ]


class Predictor(BasePredictor):
    def setup(self):
        """Initialize MediaPipe models"""
        self.face_processor = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path="thirdparty/face_landmarker_v2_with_blendshapes.task"
                ),
                output_face_blendshapes=True,
                num_faces=1,
            )
        )
        self.pose_processor = mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=2, min_detection_confidence=0.5
        )

    def predict(self, image_path: Path) -> Output:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        height, width, _ = img_np.shape

        # Convert to MediaPipe Image with explicit format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=img_np.astype(np.uint8)
        )

        # Detect features
        face_result = self.face_processor.detect(mp_image)
        pose_result = self.pose_processor.process(
            cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        )
        # Generate outputs
        results = FullBodyProcessor.process_results(
            pose_result.pose_landmarks,
            face_result.face_landmarks[0] if face_result.face_landmarks else None,
            face_result.face_blendshapes[0] if face_result.face_blendshapes else [],
            (height, width),
        )
        return Output(
            coco_keypoints=json.dumps(results["coco"], indent=2),
            facs=json.dumps(results["facs"], indent=2),
            fullbodyfacs=json.dumps(results["fullbodyfacs"], indent=2),
            vrm_skeleton=json.dumps(results["vrm"], indent=2),
            debug_image=self._create_debug_image(
                img_np, face_result, pose_result, image_path
            ),
        )

    def _create_debug_image(self, img_np, face_result, pose_result, image_path):
        """Generate annotated debug image"""
        annotated = img_np.copy()

        # Draw face landmarks
        if face_result.face_landmarks:
            landmarks = landmark_pb2.NormalizedLandmarkList()
            landmarks.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=lmk.x, y=lmk.y, z=lmk.z)
                    for lmk in face_result.face_landmarks[0]
                ]
            )
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS
            )
        # Draw pose skeleton
        if pose_result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated,
                pose_result.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
            )
        debug_path = f"/tmp/{os.path.basename(image_path)}_debug.jpg"
        Image.fromarray(annotated).save(debug_path)
        return Path(debug_path)


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    result = predictor.predict(image_path)
    print(json.dumps(json.loads(result.json()), indent=2))
