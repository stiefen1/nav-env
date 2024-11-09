"""
Utility functions for handling 2D vectors and positions, including conversions between different formats (numpy arrays, tuples, shapely Points) and calculations related to vector direction and intensity.
"""

import numpy as np
from shapely import Point
from math import pi, cos, sin, atan2

DEFAULT_ANGLE_REFERENCE: float = pi / 2.0

def assert_numpy_2d_position(position: np.ndarray) -> None:
    """
    Assert that the given numpy array is a valid 2D position.
    """
    assert_numpy_2d_vector(position)

def assert_numpy_2d_vector(vector: np.ndarray) -> None:
    """
    Assert that the given numpy array is a valid 2D vector.
    """
    assert len(vector.shape) == 1, f"Vector must be 1D numpy array, not {len(vector.shape)}D"
    assert vector.shape[0] == 2, f"Vector must have 2 coordinates (x, y), not {vector.shape[0]}"

def assert_tuple_2d_position(position: tuple) -> None:
    """
    Assert that the given tuple is a valid 2D position.
    """
    assert_tuple_2d_vector(position)

def assert_tuple_2d_vector(vector: tuple) -> None:
    """
    Assert that the given tuple is a valid 2D vector.
    """
    assert len(vector) == 2, f"Vector must have 2 coordinates (x, y), not {len(vector)}"

def convert_point_to_numpy(position: Point) -> np.ndarray:
    """
    Convert a shapely Point to a numpy array.
    """
    return np.array(position.xy).squeeze()

def convert_tuple_to_numpy(position: tuple[float, float]) -> np.ndarray:
    """
    Convert a tuple to a numpy array.
    """
    return np.array(position)

def convert_any_to_numpy(position) -> np.ndarray:
    """
    Convert any supported type (numpy array, shapely Point, tuple) to a numpy array.
    """
    if isinstance(position, np.ndarray):
        return position
    elif isinstance(position, Point):
        return convert_point_to_numpy(position)
    elif isinstance(position, tuple):
        return convert_tuple_to_numpy(position)
    else:
        raise TypeError(f"Conversion from {type(position)} is currently not handled")

def convert_point_to_tuple(position: Point) -> tuple:
    """
    Convert a shapely Point to a tuple.
    """
    return tuple([val[0] for val in position.xy])

def convert_numpy_to_tuple(position: np.ndarray) -> tuple:
    """
    Convert a numpy array to a tuple.
    """
    return tuple(position.tolist())

def convert_any_to_tuple(position) -> tuple:
    """
    Convert any supported type (tuple, shapely Point, numpy array) to a tuple.
    """
    if isinstance(position, tuple):
        return position
    elif isinstance(position, Point):
        return convert_point_to_tuple(position)
    elif isinstance(position, np.ndarray):
        return convert_numpy_to_tuple(position)
    else:
        raise TypeError(f"Conversion from {type(position)} is currently not handled")

def get_direction_intensity_from_vector(vector: np.ndarray | tuple, angle_ref: float = DEFAULT_ANGLE_REFERENCE) -> tuple[float, float]:
    """
    Calculate the direction and intensity (magnitude) of a vector.
    """
    vector: np.ndarray = convert_any_to_numpy(vector)
    intensity = float(np.linalg.norm(vector))
    direction = atan2(vector[1], vector[0]) - angle_ref
    return direction, intensity

def get_vector_from_direction_intensity(direction: float, intensity: float, angle_ref: float = DEFAULT_ANGLE_REFERENCE) -> np.ndarray:
    """
    Calculate the vector components from a given direction and intensity.
    """
    assert intensity > 0, "Norm of the desired vector must be greater than zero"
    return cos(direction + angle_ref) * intensity, sin(direction + angle_ref) * intensity

