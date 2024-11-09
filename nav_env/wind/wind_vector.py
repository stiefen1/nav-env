"""
Classes for representing wind vectors and collections of wind vectors, including methods for parsing input data and calculating properties such as direction and intensity.
"""

import numpy as np
from shapely import Point, Polygon
from nav_env.geometry.utils import *
from nav_env.geometry.vector import Vector


class WindVector(Vector):
    """
    Represents a wind vector with position and velocity.
    """
    def __init__(self, position: np.ndarray | Point | tuple[float, float], *args, **kwargs):
        super().__init__(position, *args, **kwargs)

    @property
    def speed(self) -> float:
        return self.intensity
    
    @property
    def velocity(self) -> tuple[float, float]:
        return self.vector

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
        return [vec.vy for vec in self._wind_vectors]

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
    w11 = WindVector(p_numpy, intensity=speed, direction=direction)
    w12 = WindVector(p_shapely, intensity=speed, direction=direction)
    w13 = WindVector(p_tuple, intensity=speed, direction=direction)
    w21 = WindVector(p_numpy, vector=velocity)
    w22 = WindVector(p_numpy, vector=velocity)
    w23 = WindVector(p_tuple, vector=velocity)
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