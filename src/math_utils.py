import numpy as np


# Check calc_angle_between_three_2d_points([1,1], [2, 2], [3, 1]) == 90 - Good :)
def calc_angle_between_three_2d_points(a: list[float], b: list[float], c: list[float]) -> float:
    """
    Calculate the vertex angle between 3 points in 3D space

    :return: The angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
