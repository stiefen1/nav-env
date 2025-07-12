import numpy as np

def wrap_min_max(x: float | np.ndarray, x_min: float | np.ndarray, x_max: float | np.ndarray) -> float | np.ndarray:
    """Wraps input x to [x_min, x_max)

    Args:
        x (float or np.ndarray): Unwrapped value
        x_min (float or np.ndarray): Minimum value
        x_max (float or np.ndarray): Maximum value

    Returns:
        float or np.ndarray: Wrapped value
    """
    if isinstance(x, np.ndarray):
        return x_min + np.mod(x - x_min, x_max - x_min)
    else:
        return x_min + (x - x_min) % (x_max - x_min)


def wrap_angle_to_pmpi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wraps input angle to [-pi, pi)

    Args:
        angle (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle
    """
    if isinstance(angle, np.ndarray):
        return wrap_min_max(angle, -np.pi * np.ones(angle.size), np.pi * np.ones(angle.size))
    else:
        return wrap_min_max(angle, -np.pi, np.pi)
    
def wrap_angle_to_pmpi_degrees(angle: float | np.ndarray) -> float | np.ndarray:
    """Wraps input angle to [-180, 180)

    Args:
        angle (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle
    """
    if isinstance(angle, np.ndarray):
        return wrap_min_max(angle, -180 * np.ones(angle.size), 180 * np.ones(angle.size))
    else:
        return wrap_min_max(angle, -180, 180)