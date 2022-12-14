import os

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from constants import mp_holistic, index_to_body_mapping, VERTEX_ANGLE_NAMES
from math_utils import calc_angle_between_three_2d_points


def get_valid_pose_folder_names(folder_dir: str, must_start_with_pose: bool = True) -> list[str]:
    """
    Retrieve folder names which start with pose to create training dataset
    :param folder_dir: Directory to search within
    :param must_start_with_pose: Whether the folders are required to start with 'pose'
    :return: List of valid folders
    """
    folders = os.listdir(folder_dir)

    # Get valid folders
    valid_folders = []
    for subfolder in folders:
        bool_clause = must_start_with_pose == subfolder.startswith("pose")
        if bool_clause and os.path.isdir(os.path.join(folder_dir, subfolder)):
            valid_folders.append(subfolder)

    return valid_folders


def generate_pose_landmarks(
    input_img: np.array, min_detection_confidence: float = 0.75, model_complexity: int = 2
) -> pd.DataFrame:
    """
    Using the Media Pipes Holistic mode, generate the pose estimation for a given image and package
    them neatly within a dataframe

    :param input_img: Input image
    :param min_detection_confidence: Minimum detection confidence value ([0.0, 1.0]) for person
        detection to be considered successful. See details in
        https://solutions.mediapipe.dev/holistic#min_detection_confidence.
    :param model_complexity: Complexity of the pose landmark model: 0, 1 or 2. See
        details in https://solutions.mediapipe.dev/holistic#model_complexity.
    :return: DataFrame of the pose information containing the following columns:
         ["pose_landmark", "x", "y", "z", "visibility"]
    """
    with mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=min_detection_confidence,
        model_complexity=model_complexity,
    ) as pose:
        input_results = pose.process(input_img)

        landmark_body_arr = []
        for i, landmark in enumerate(input_results.pose_world_landmarks.landmark):
            body_part = index_to_body_mapping[i]
            landmark_body_arr.append(
                [body_part, landmark.x, landmark.y, landmark.z, landmark.visibility]
            )

        columns = ["pose_landmark", "x", "y", "z", "visibility"]
        input_landmark_pose_df = pd.DataFrame(data=landmark_body_arr, columns=columns)

    return input_landmark_pose_df


def get_bodypart_angle_df(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """Given the landmarks df, extract and return the unique vertex angle features into a
    packaged dataframe
    """
    # lowercase all poses
    landmarks_df["pose_landmark"] = landmarks_df["pose_landmark"].str.lower()

    # Populate for each
    angles_dict = get_vertex_angles_dict(landmarks_df)
    df = pd.DataFrame(columns=VERTEX_ANGLE_NAMES)
    df = df.append(angles_dict, ignore_index=True)

    return df


def estimate_line_vertex_angles(
    landmarks_df: pd.DataFrame, pos_1: str, vertex_pos: str, pos_3: str
) -> float:
    """
    Given 3 input pose position names, extract the points and calculate the vertex angle between
     the 3 points in 3D space

    :param landmarks_df: The dataframe containing pose landmarks. Includes the following columns:
        ["pose_landmark", "x", "y", "z", "visibility"]
    :param pos_1: First position name
    :param vertex_pos: Vertex position name (point connecting pos_1 and pos_2
    :param pos_3: Third position name
    :return: The angle in degrees
    """
    p1 = (
        landmarks_df[landmarks_df["pose_landmark"] == pos_1]
        .iloc[0][["x", "y", "z"]]
        .values.tolist()
    )
    p2 = (
        landmarks_df[landmarks_df["pose_landmark"] == vertex_pos]
        .iloc[0][["x", "y", "z"]]
        .values.tolist()
    )
    p3 = (
        landmarks_df[landmarks_df["pose_landmark"] == pos_3]
        .iloc[0][["x", "y", "z"]]
        .values.tolist()
    )

    angle_degrees = calc_angle_between_three_2d_points(p1, p2, p3)
    assert angle_degrees <= 360

    return angle_degrees


def get_vertex_angles_dict(landmarks_df: pd.DataFrame) -> dict:
    """
    Extract all the vertex angle features from the landmarks

    :param landmarks_df: The dataframe containing pose landmarks. Includes the following columns:
        ["pose_landmark", "x", "y", "z", "visibility"]
    :return: Dictionary containing all the vertex angle features
    """
    angles_dict = {}
    angles_dict["Left_armpit_angle"] = estimate_line_vertex_angles(
        landmarks_df, "left_shoulder", "left_elbow", "left_hip"
    )
    angles_dict["Right_armpit_angle"] = estimate_line_vertex_angles(
        landmarks_df, "right_shoulder", "right_elbow", "right_hip"
    )
    angles_dict["Left_shoulder_angle"] = estimate_line_vertex_angles(
        landmarks_df, "left_shoulder", "right_shoulder", "left_hip"
    )
    angles_dict["Right_shoulder_angle"] = estimate_line_vertex_angles(
        landmarks_df, "right_shoulder", "left_shoulder", "right_hip"
    )
    angles_dict["Left_elbow_angle"] = estimate_line_vertex_angles(
        landmarks_df, "left_elbow", "left_shoulder", "left_wrist"
    )
    angles_dict["Right_elbow_angle"] = estimate_line_vertex_angles(
        landmarks_df, "right_elbow", "right_shoulder", "right_wrist"
    )
    angles_dict["Left_hip_angle"] = estimate_line_vertex_angles(
        landmarks_df, "left_hip", "right_hip", "left_shoulder"
    )
    angles_dict["Right_hip_angle"] = estimate_line_vertex_angles(
        landmarks_df, "right_hip", "left_hip", "right_shoulder"
    )
    angles_dict["Left_groin_angle"] = estimate_line_vertex_angles(
        landmarks_df, "left_hip", "left_knee", "left_ankle"
    )
    angles_dict["Right_groin_angle"] = estimate_line_vertex_angles(
        landmarks_df, "right_hip", "right_knee", "right_ankle"
    )
    angles_dict["Left_knee_angle"] = estimate_line_vertex_angles(
        landmarks_df, "left_knee", "left_ankle", "left_hip"
    )
    angles_dict["Right_knee_angle"] = estimate_line_vertex_angles(
        landmarks_df, "right_knee", "right_ankle", "right_hip"
    )

    return angles_dict


def fit_knn_classifier(
    df: pd.DataFrame, target_col_name: str = "pose_type", n_neighbors: int = 3
) -> KNeighborsClassifier:
    """Fit the k-nearest neighbors classifier to the given PCA data"""
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(df.drop(columns=target_col_name), df[target_col_name])

    return knn_classifier
