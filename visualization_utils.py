import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from constants import mp_drawing, mp_pose


def plot_2d_pca_figure(df: pd.DataFrame, point_to_add: np.array = None) -> plt.axes.Axes:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('PCA of Positions', fontsize=20)

    poses = df["pose_type"].unique().tolist()
    colors = ['r', 'g', 'b', 'y', 'm', 'c']

    for pose, color in zip(poses, colors):
        idx_to_display = df['pose_type'] == pose
        ax.scatter(
            df.loc[idx_to_display, 'principal component 1'],
            df.loc[idx_to_display, 'principal component 2'],
            c=color,
            alpha=0.7
        )

    if point_to_add is not None:
        ax.scatter(
            point_to_add[0],
            point_to_add[1],
            c="k",
            alpha=1.0,
            marker="X",
            s=150,
        )
        ax.legend(poses + ["Unlabeled Pose"], loc='upper left')
    else:
        ax.legend(poses, loc='upper left')

    ax.grid()
    plt.show()

    return ax


def _normalize_color(color: list[float]):
    return tuple(v / 255. for v in color)


def read_and_display_image(img_path: str, true_pose: str) -> np.array:
    input_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.title(f"True Pose: {true_pose}", fontsize=18)
    plt.imshow(input_img)
    plt.show()

    return input_img


def flip_image_horizontally(img_rgb: np.array) -> np.array:
    image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    flip_image = cv2.flip(image, 1)

    flip_rgb_image = cv2.cvtColor(flip_image, cv2.COLOR_BGR2RGB)
    return flip_rgb_image


def custom_plot_landmarks_from_df(
        landmark_df: pd.DataFrame,
        ax: plt.axes.Axes = None,
        add_connections: bool = True,
        landmark_drawing_spec: mp_drawing.DrawingSpec = mp_drawing.DrawingSpec(
            color=mp_drawing.RED_COLOR, thickness=5),
        connection_drawing_spec: mp_drawing.DrawingSpec = mp_drawing.DrawingSpec(
            color=mp_drawing.BLACK_COLOR, thickness=5),
        visibility_threshold: float = 0.5,
        presence_threshold: float = 0.5,
) -> plt.axes.Axes:
    """Plot the landmarks and the connections in matplotlib 3d.
    Args:
    landmark_list: A normalized landmark list proto message to be plotted.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected.
    landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
      drawing settings such as color and line thickness.
    connection_drawing_spec: A DrawingSpec object that specifies the
      connections' drawing settings such as color and line thickness.
    elevation: The elevation from which to view the plot.
    azimuth: the azimuth angle to rotate the plot.
    Raises:
    ValueError: If any connections contain invalid landmark index.
    """
    if landmark_df is None:
        return

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=10, azim=10)

    plotted_landmarks = {}
    for i_row, landmark_dict in landmark_df.iterrows():
        if (('visibility' in landmark_dict) and (
                landmark_dict["visibility"] < visibility_threshold)) or \
                (('presence' in landmark_dict) and (
                        landmark_dict['presence'] < presence_threshold)):
            continue

        ax.scatter3D(
            xs=[-landmark_dict['z']],
            ys=[landmark_dict['x']],
            zs=[-landmark_dict['y']],
            color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness
        )
        plotted_landmarks[i_row] = (-landmark_dict['z'], landmark_dict['x'], -landmark_dict['y'])

    if add_connections:
        num_landmarks = landmark_df.shape[0]

        # Draws the connections if the start and end landmarks are both visible.
        connections = mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f'Landmark index is out of range. Invalid connection from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness
                )

    return ax
