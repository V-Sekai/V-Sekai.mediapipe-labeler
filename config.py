COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "brow_inner_left", "brow_inner_right", "brow_outer_left",
    "brow_outer_right", "lid_upper_left", "lid_upper_right",
    "lid_lower_left", "lid_lower_right", "lip_upper", "lip_lower",
    "lip_corner_left", "lip_corner_right"
]

HAND_MAPPINGS = {
    "left": {
        1: "leftThumbMetacarpal", 2: "leftThumbProximal", 3: "leftThumbDistal",
        5: "leftIndexProximal", 6: "leftIndexIntermediate", 7: "leftIndexDistal",
        9: "leftMiddleProximal", 10: "leftMiddleIntermediate", 11: "leftMiddleDistal",
        13: "leftRingProximal", 14: "leftRingIntermediate", 15: "leftRingDistal",
        17: "leftLittleProximal", 18: "leftLittleIntermediate", 19: "leftLittleDistal"
    },
    "right": {
        1: "rightThumbMetacarpal", 2: "rightThumbProximal", 3: "rightThumbDistal",
        5: "rightIndexProximal", 6: "rightIndexIntermediate", 7: "rightIndexDistal",
        9: "rightMiddleProximal", 10: "rightMiddleIntermediate", 11: "rightMiddleDistal",
        13: "rightRingProximal", 14: "rightRingIntermediate", 15: "rightRingDistal",
        17: "rightLittleProximal", 18: "rightLittleIntermediate", 19: "rightLittleDistal"
    }
}

FACS_CONFIG = {
    "AU_MAPPING": {
        "AU1": ["browInnerUp"],
        "AU2": ["browOuterUpLeft", "browOuterUpRight"],
        "AU4": ["browDownLeft", "browDownRight"],
        "AU5": ["eyeBlinkLeft", "eyeBlinkRight"],
        "AU6": ["eyeSquintLeft", "eyeSquintRight"],
        "AU9": ["noseSneerLeft", "noseSneerRight"],
        "AU12": ["mouthSmileLeft", "mouthSmileRight"],
        "AU25": ["jawOpen", "mouthStretch"]
    },
    "LANDMARK_MAPPING": {
        "AU1": {"inner_brow": [105, 334]},
        "AU2": {"outer_brow": [46, 276]},
        "AU4": {"brow_depressor": [55, 285]},
        "AU5": {"upper_lid": [159, 386]},
        "AU6": {"cheek_raiser": [112, 342]},
        "AU12": {"lip_corner": [61, 291]},
        "AU25": {"jaw_open": [200, 17]}
    }
}


LEFT_HAND_VRM_MAPPING = {
    1: "leftThumbMetacarpal",
    2: "leftThumbProximal",
    3: "leftThumbDistal",
    5: "leftIndexProximal",
    6: "leftIndexIntermediate",
    7: "leftIndexDistal",
    9: "leftMiddleProximal",
    10: "leftMiddleIntermediate",
    11: "leftMiddleDistal",
    13: "leftRingProximal",
    14: "leftRingIntermediate",
    15: "leftRingDistal",
    17: "leftLittleProximal",
    18: "leftLittleIntermediate",
    19: "leftLittleDistal",
}

RIGHT_HAND_VRM_MAPPING = {
    1: "rightThumbMetacarpal",
    2: "rightThumbProximal",
    3: "rightThumbDistal",
    5: "rightIndexProximal",
    6: "rightIndexIntermediate",
    7: "rightIndexDistal",
    9: "rightMiddleProximal",
    10: "rightMiddleIntermediate",
    11: "rightMiddleDistal",
    13: "rightRingProximal",
    14: "rightRingIntermediate",
    15: "rightRingDistal",
    17: "rightLittleProximal",
    18: "rightLittleIntermediate",
    19: "rightLittleDistal",
}
