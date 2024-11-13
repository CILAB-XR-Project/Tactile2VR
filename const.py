
import numpy as np

TACTILE_SIZE = [32, 22]

ACTIVITY_LIST = [
    "Squat",
    "Lunge",
    "Jump",
    "Stepper",
    "Walking",  # walking should be in front of other walking variants
    "InPlaceWalking",
    "SideWalking",
    "BackwardWalking",
]

COCO_19_KEYPOINTS = [
    'nose', #0
    'left_shoulder', #1
    'right_shoulder', #2
    'left_elbow', #3
    'right_elbow',#4
    'left_wrist',#5
    'right_wrist',#6
    'left_hip_extra',#7
    'right_hip_extra',#8
    'left_knee',#9
    'right_knee',#10
    'left_ankle',#11
    'right_ankle',#12
    'left_bigtoe',#13
    'left_smalltoe',#14
    'left_heel',#15
    'right_bigtoe',#16
    'right_smalltoe',#17
    'right_heel',#18
]

KEYPOINTS_NAMES_TO_INDEX = {
    'nose':  0,
    'left_shoulder':  1,
    'right_shoulder':  2,
    'left_elbow':  3,
    'right_elbow':  4,
    'left_wrist':  5,
    'right_wrist':  6,
    'left_hip_extra':  7,
    'right_hip_extra':  8,
    'left_knee':  9,
    'right_knee':  10,
    'left_ankle':  11,
    'right_ankle':  12,
    'left_bigtoe':  13,
    'left_smalltoe':  14,
    'left_heel':  15,
    'right_bigtoe':  16,
    'right_smalltoe':  17,
    'right_heel':  18,
}

VR_INDEXS = [
    KEYPOINTS_NAMES_TO_INDEX["nose"],
    KEYPOINTS_NAMES_TO_INDEX["left_wrist"],
    KEYPOINTS_NAMES_TO_INDEX["right_wrist"]
]

HEAD_INDEXS = [0]
SHOULDER_INDEXS = [1, 2]
ELBOW_INDEXS = [3, 4]
WRIST_INDEXS = [5, 6]
HIP_INDEXS = [7, 8]
KNEE_INDEXS = [9, 10]
ANKLE_INDEXS = [11, 12]
FEET_INDEXS = [13, 14, 15, 16, 17, 18]


BODY_18_PAIRS = [
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (1, 7),
    (2, 8),
    (7, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (10, 12),
    (11, 15),
    (12, 18),
    (15, 13),
    (15, 14),
    (18, 16),
    (18, 17),
]

BODY_25_color = np.array([[255, 0, 0], [255, 51, 0], [255, 102, 0], [255, 153, 0], [255, 204, 0], [255, 255, 0], [204, 255, 0]
                         , [153, 255, 0], [102, 255, 0], [51, 255, 0], [0, 255, 0], [0, 255, 51], [0, 255, 102], [0,255,153]
                         , [0, 255, 204], [0, 255, 255], [0, 204, 255], [0, 153, 255], [0, 102, 255], [0, 53, 255], [0, 0, 255]
                         , [53, 0, 255], [102, 0, 255], [153, 0, 255], [204, 0, 255], [255, 0, 255]])
