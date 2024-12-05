import numpy as np
from shapely import Point
from nav_env.geometry.utils import *
import matplotlib.pyplot as plt

class Vector:
    """
    Represents a vector at a give position.
    """
    def __init__(self, position: np.ndarray | Point | tuple[float, float], *args, **kwargs):
        self._parse_position(position)
        self._parse_args(*args)
        self._parse_kwargs(**kwargs)

    def quiver(self, ax=None, **kwargs):
        """
        Plot the vector as a quiver.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.quiver(self.x, self.y, self.vx, self.vy, **kwargs)
        return ax

    @property
    def x(self) -> float:
        return self._position[0]
    
    @property
    def y(self) -> float:
        return self._position[1]
    
    @property
    def vx(self) -> float:
        return self._vector[0]
    
    @property
    def vy(self) -> float:
        return self._vector[1]
    
    @property
    def position(self) -> tuple[float, float]:
        return self._position

    @property
    def vector(self) -> tuple[float, float]:
        return self._vector
    
    @property
    def direction(self) -> float:
        return get_direction_intensity_from_vector(self._vector)[0]
    
    @property
    def intensity(self) -> float:
        return float(get_direction_intensity_from_vector(self._vector)[1])

    def _parse_position(self, position: np.ndarray | Point | tuple[float, float]):
        """
        Parse and validate the position input.
        """
        position: tuple = convert_any_to_tuple(position)
        assert_tuple_2d_position(position)
        self._position = position

    def _parse_args(self, *args):
        """
        Parse additional positional arguments.
        """
        pass

    def _parse_kwargs(self, **kwargs):
        """
        Parse keyword arguments for vector or intensity and direction.
        """
        keys = kwargs.keys()
        if 'intensity' in keys and 'direction' in keys:
            vector: tuple = get_vector_from_direction_intensity(kwargs['direction'], kwargs['intensity'])
        elif 'vector' in keys:
            vector: tuple = convert_any_to_tuple(kwargs['vector'])
        else:
            self._vector = None
            return 
        
        assert_tuple_2d_vector(vector)
        self._vector = vector

    def __str__(self) -> str:
        return f"WindVector Object : Pos: {", ".join(f"{p:.3f}" for p in self.position)} Vel: {", ".join(f"{p:.3f}" for p in self.vector)}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __mul__(self, other: float) -> "Vector":
        return Vector(self.position, vector=(self.vx * other, self.vy * other))
    
    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.position, vector=(self.vx + other.vx, self.vy + other.vy))
