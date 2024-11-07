"""
Classes for representing wind vectors and collections of wind vectors, including methods for parsing input data and calculating properties such as direction and intensity.
"""

import numpy as np
from shapely import Point, Polygon
from .utils import *

class WindVector:
    """
    Represents a wind vector with position and velocity.
    """
    def __init__(self, position: np.ndarray | Point | tuple[float, float], *args, **kwargs):
        self._parse_position(position)
        self._parse_args(*args)
        self._parse_kwargs(**kwargs)

    @property
    def x(self) -> float:
        return self._position[0]
    
    @property
    def y(self) -> float:
        return self._position[1]
    
    @property
    def vx(self) -> float:
        return self._velocity[0]
    
    @property
    def vy(self) -> float:
        return self._velocity[1]
    
    @property
    def position(self) -> tuple[float, float]:
        return self._position

    @property
    def velocity(self) -> tuple[float, float]:
        return self._velocity
    
    @property
    def direction(self) -> float:
        return get_direction_intensity_from_vector(self._velocity)[0]
    
    @property
    def intensity(self) -> float:
        return float(get_direction_intensity_from_vector(self._velocity)[1])

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
        Parse keyword arguments for velocity or speed and direction.
        """
        keys = kwargs.keys()
        if 'speed' in keys and 'direction' in keys:
            velocity: tuple = get_vector_from_direction_intensity(kwargs['direction'], kwargs['speed'])
        elif 'velocity' in keys:
            velocity: tuple = convert_any_to_tuple(kwargs['velocity'])
        else:
            raise KeyError("You must provide either two float (speed, direction) or a numpy array representing wind velocity vector")
        
        assert_tuple_2d_vector(velocity)
        self._velocity = velocity

    def __str__(self) -> str:
        return f"WindVector Object : Pos: {self.position} Vel: {self.velocity}"

class WindVectorCollection:
    """
    Represents a collection of WindVector objects and provides methods to compute properties of the collection.
    """
    def __init__(self, wind_vectors: list[WindVector]) -> None:
        self._wind_vectors = wind_vectors
        self._compute_domain()

    def _compute_domain(self) -> Polygon:
        """
        Compute the convex hull of the positions of the wind vectors.
        """
        vertices = [vec.position for vec in self._wind_vectors]
        self._domain = Polygon(vertices).convex_hull

    def __iter__(self):
        for vec in self._wind_vectors:
            yield vec

    def __str__(self) -> str:
        return f"WindVectorCollection Object : {len(self._wind_vectors)} WindVector objects"
    
    def __getitem__(self, index: int) -> WindVector:
        return self._wind_vectors[index]

    @property
    def positions(self) -> tuple:
        return [vec.position for vec in self._wind_vectors]
    
    @property
    def positions_x(self) -> tuple:
        return [vec.x for vec in self._wind_vectors]
    
    @property
    def positions_y(self) -> tuple:
        return [vec.y for vec in self._wind_vectors]
    
    @property
    def velocities(self) -> tuple:
        return [vec.velocity for vec in self._wind_vectors]
    
    @property
    def velocities_x(self) -> tuple:
        return [vec.vx for vec in self._wind_vectors]
    
    @property
    def velocities_y(self) -> tuple:
        return [vec.vx for vec in self._wind_vectors]

    @property
    def domain(self) -> Polygon:
        return self._domain

    @property
    def y_max(self) -> float:
        return max(self.positions_y)
    
    @property
    def y_min(self) -> float:
        return min(self.positions_y)
    
    @property
    def x_max(self) -> float:
        return max(self.positions_x)
    
    @property
    def x_min(self) -> float:
        return min(self.positions_x)
    
def test():
    """
    Test function for WindVector and WindVectorCollection classes.
    """
    x, y = 3., 2.
    p_shapely = Point([x, y])
    p_numpy = np.array([x, y])
    p_tuple = (x, y)

    speed, direction = 3., 0.352
    velocity = get_vector_from_direction_intensity(direction, speed)
    
    # Check all the different ways of instantiating a WindVector
    w11 = WindVector(p_numpy, speed=speed, direction=direction)
    w12 = WindVector(p_shapely, speed=speed, direction=direction)
    w13 = WindVector(p_tuple, speed=speed, direction=direction)
    w21 = WindVector(p_numpy, velocity=velocity)
    w22 = WindVector(p_numpy, velocity=velocity)
    w23 = WindVector(p_tuple, velocity=velocity)
    ws = WindVectorCollection([w11, w12, w13, w21, w22, w23])

    # Compute error in position / velocity results
    dp, dv, count = 0., 0., 0
    for i, wi in enumerate(ws):
        for j in range(i):
            count += 1
            wj = ws[j]
            dp += ((wi.x - wj.x) ** 2 + (wi.y - wj.y) ** 2) ** 0.5
            dv += ((wi.x - wj.x) ** 2 + (wi.y - wj.y) ** 2) ** 0.5

    assert dp == 0 and dv == 0
    print(f"Successfully finished wind_vector test with output {dp}, {dv}, {count}")

if __name__ == "__main__":
    test()