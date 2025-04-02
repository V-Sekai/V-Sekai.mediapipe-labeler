from config import COCO_KEYPOINT_NAMES, HAND_MAPPINGS, FACS_CONFIG, LEFT_HAND_VRM_MAPPING, RIGHT_HAND_VRM_MAPPING
from typing import List, Dict, Any

class FullBodyProcessor:
    @staticmethod
    def process_results(pose, face, blendshapes, left_hand, right_hand, image_size):
        return {
            "coco": FullBodyProcessor.create_coco_output(pose, face, image_size),
            "facs": FullBodyProcessor.create_facs_output(blendshapes),
            "fullbodyfacs": FullBodyProcessor.create_fullbodyfacs(
                pose, face, image_size
            ),
            "hands": FullBodyProcessor.process_hands(left_hand, right_hand),
        }

    @staticmethod
    def create_coco_output(pose, face, image_size):
        height, width = image_size
        keypoints = []
        num_visible = 0

        if pose:
            for idx in range(17):
                if idx < len(pose.landmark):
                    lmk = pose.landmark[idx]
                    vis = lmk.visibility
                    if vis < 0.1:  # Filter low-confidence keypoints
                        keypoints += [0.0, 0.0, 0]
                    else:
                        x = lmk.x * width
                        y = lmk.y * height
                        vis_flag = 2 if vis > 0.5 else 1
                        keypoints += [x, y, vis_flag]
                        num_visible += 1 if vis_flag == 2 else 0
                else:
                    keypoints += [0.0, 0.0, 0]

        # Updated facial indices based on MediaPipe's face landmark model
        facial_indices = [
            105,  # brow_inner_left
            334,  # brow_inner_right
            46,  # brow_outer_left
            276,  # brow_outer_right
            159,  # lid_upper_left
            386,  # lid_upper_right
            145,  # lid_lower_left
            374,  # lid_lower_right
            13,  # lip_upper
            14,  # lip_lower
            61,  # lip_corner_left
            291,  # lip_corner_right
        ]

        if face:
            for idx in facial_indices:
                if idx < len(face):
                    lmk = face[idx]
                    keypoints += [lmk.x * width, lmk.y * height, 2]
                    num_visible += 1
                else:
                    keypoints += [0.0, 0.0, 0]

        # Improved neck position calculation
        if pose and len(pose.landmark) > 12:
            nose_lmk = pose.landmark[0]
            left_shoulder = pose.landmark[11]
            right_shoulder = pose.landmark[12]

            # Only adjust if both shoulders are visible
            if left_shoulder.visibility >= 0.5 and right_shoulder.visibility >= 0.5:
                neck_x = (left_shoulder.x + right_shoulder.x) / 2 * width
                keypoints[0] = (nose_lmk.x * width + neck_x) / 2
            else:
                keypoints[0] = nose_lmk.x * width
            keypoints[1] = nose_lmk.y * height

        return {
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": num_visible,
                    "bbox": FullBodyProcessor.calculate_bbox(keypoints),
                    "area": width * height,
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": COCO_KEYPOINT_NAMES,
                    "skeleton": [
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
                        [17, 18],
                        [18, 0],
                        [19, 17],
                        [20, 18],
                        [21, 17],
                        [22, 18],
                        [23, 21],
                        [24, 22],
                        [25, 27],
                        [26, 28],
                        [27, 25],
                        [28, 25],
                    ],
                }
            ],
        }

    @staticmethod
    def create_facs_output(blendshapes):
        au_scores = {}
        blendshape_dict = {}
        for bs in blendshapes:
            if hasattr(bs, "category_name") and hasattr(bs, "score"):
                blendshape_dict[bs.category_name] = bs.score

        FACS_AU_MAPPING = {
            "AU1": ["browInnerUp"],
            "AU2": ["browOuterUpLeft", "browOuterUpRight"],
            "AU4": ["browDownLeft", "browDownRight"],
            "AU5": ["eyeBlinkLeft", "eyeBlinkRight"],
            "AU6": ["eyeSquintLeft", "eyeSquintRight"],
            "AU9": ["noseSneerLeft", "noseSneerRight"],
            "AU12": ["mouthSmileLeft", "mouthSmileRight"],
            "AU25": ["jawOpen", "mouthStretch"],
        }

        for au, components in FACS_AU_MAPPING.items():
            scores = [blendshape_dict.get(name, 0.0) for name in components]
            au_scores[au] = sum(scores) / len(scores) if scores else 0.0

        return {
            "AUs": au_scores,
            "blendshapes": [
                {"name": bs.category_name, "score": float(bs.score)}
                for bs in blendshapes
                if hasattr(bs, "category_name") and hasattr(bs, "score")
            ],
        }

    @staticmethod
    def create_fullbodyfacs(pose, face, image_size):
        keypoints = []
        height, width = image_size
        left_shoulder = right_shoulder = None

        if pose:
            for idx, lmk in enumerate(pose.landmark):
                keypoints.append(
                    {
                        "id": idx,
                        "name": f"body_{idx}",
                        "position": [lmk.x * width, lmk.y * height, lmk.z * width],
                        "parent": FullBodyProcessor.get_parent(idx),
                        "visibility": lmk.visibility,
                    }
                )

                if idx == 11:
                    left_shoulder = lmk
                elif idx == 12:
                    right_shoulder = lmk

            # Improved neck calculation with visibility check
            if left_shoulder and right_shoulder:
                if left_shoulder.visibility >= 0.5 and right_shoulder.visibility >= 0.5:
                    neck_x = (left_shoulder.x + right_shoulder.x) / 2 * width
                    neck_y = (left_shoulder.y + right_shoulder.y) / 2 * height
                    neck_z = (left_shoulder.z + right_shoulder.z) / 2 * width
                    keypoints.append(
                        {
                            "id": 33,
                            "name": "neck",
                            "position": [neck_x, neck_y, neck_z],
                            "parent": 0,
                            "visibility": min(
                                left_shoulder.visibility, right_shoulder.visibility
                            ),
                        }
                    )

        facial_map = {
            105: 34,  # brow_inner_left
            334: 35,  # brow_inner_right
            46: 36,  # brow_outer_left
            276: 37,  # brow_outer_right
            159: 38,  # lid_upper_left
            386: 39,  # lid_upper_right
            145: 40,  # lid_lower_left
            374: 41,  # lid_lower_right
            13: 42,  # lip_upper
            14: 43,  # lip_lower
            61: 44,  # lip_corner_left
            291: 45,  # lip_corner_right
        }

        if face:
            for mp_idx, facs_id in facial_map.items():
                if mp_idx < len(face):
                    lmk = face[mp_idx]
                    keypoints.append(
                        {
                            "id": facs_id,
                            "name": COCO_KEYPOINT_NAMES[facs_id - 17],
                            "position": [lmk.x * width, lmk.y * height, lmk.z * width],
                            "parent": 0,
                            "visibility": 1.0,
                        }
                    )

        return {"keypoints": keypoints}

    @staticmethod
    def calculate_bbox(keypoints):
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

    @staticmethod
    def process_hands(left_hand, right_hand):
        def process_single_hand(hand, is_left=True):
            if not hand:
                return []
            return [
                {
                    "index": idx,
                    "x": lmk.x,
                    "y": lmk.y,
                    "name": ("left_wrist" if is_left else "right_wrist")
                    if idx == 0
                    else (
                        LEFT_HAND_VRM_MAPPING if is_left else RIGHT_HAND_VRM_MAPPING
                    ).get(idx, None),
                }
                for idx, lmk in enumerate(hand)
            ]

        return {
            "left": process_single_hand(left_hand, True),
            "right": process_single_hand(right_hand, False),
        }

    @staticmethod
    def get_parent(idx):
        return {
            11: 33,
            12: 33,
            33: 0,
            13: 11,
            14: 12,
            15: 13,
            16: 14,
            23: 11,
            24: 12,
            25: 23,
            26: 24,
            27: 25,
            28: 26,
            34: 35,
            35: 0,
            36: 34,
            37: 35,
            38: 34,
            39: 35,
            40: 38,
            41: 39,
            42: 44,
            43: 45,
            44: 42,
            45: 42,
        }.get(idx, -1)
