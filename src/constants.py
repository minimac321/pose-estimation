import mediapipe as mp

# Media Pipe Constants
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

POSE_LANDMARKS = [m.name for m in mp_pose.PoseLandmark]
index_to_body_mapping = {i: body_part.lower() for i, body_part in enumerate(POSE_LANDMARKS)}

VERTEX_ANGLE_NAMES = [
    "Left_armpit_angle",
    "Right_armpit_angle",
    "Left_shoulder_angle",
    "Right_shoulder_angle",
    "Left_elbow_angle",
    "Right_elbow_angle",
    "Left_hip_angle",
    "Right_hip_angle",
    "Left_groin_angle",
    "Right_groin_angle",
    "Left_knee_angle",
    "Right_knee_angle",
]
