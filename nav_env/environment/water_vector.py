from nav_env.geometry.vector import Vector
import numpy as np
from shapely import Point


class WaterVector(Vector):
    """
    Represents a water vector with position and velocity.
    """
    def __init__(self, position: np.ndarray | Point | tuple[float, float], *args, **kwargs):
        super().__init__(position, *args, **kwargs)

    @property
    def speed(self) -> float:
        return self.intensity
    
    @property
    def velocity(self) -> tuple[float, float]:
        return self.vector