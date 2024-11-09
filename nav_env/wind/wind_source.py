"""
Abstract base class and concrete implementations for wind sources, including methods for interpolating wind data and plotting wind fields.
"""

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from shapely import Point, Polygon
from nav_env.wind.wind_vector import WindVector, WindVectorCollection
from nav_env.geometry.utils import *
from nav_env.geometry.vector_source import VectorSource
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

class CallableWindSource(VectorSource):
    """
    Wind source defined by a callable function.
    """
    def __init__(self, wind_function: Callable[[tuple[float, float]], WindVector], domain: Polygon = None):
        super().__init__(domain)
        self._wind_function = wind_function

    def __get_vector__(self, x: float, y: float, *args, **kwargs) -> WindVector:
        return self._wind_function(x, y, *args, **kwargs)
    
    def plot(self, lim: tuple[tuple, tuple], nx=30, ny=30, *args, **kwargs):
        self.__plot__(lim, nx, ny, *args, **kwargs)

class MeasuredWindSource(VectorSource):
    """
    Wind source based on measured wind vectors.
    """
    def __init__(self, measurements: list[WindVector]):
        self._collection = WindVectorCollection(measurements)
        super().__init__(self._collection.domain)

    def _interpolate(self, x: float, y: float, *args, **kwargs) -> WindVector:
        """
        Interpolate the wind vector at a given position using measured data.
        """
        assert Point([x, y]).within(self.domain), f"Point ({x:.1f},{y:.1f}) is out of the domain"
        vx = griddata(self._collection.positions, self._collection.velocities_x, (x, y), *args, **kwargs)
        vy = griddata(self._collection.positions, self._collection.velocities_y, (x, y), *args, **kwargs)
        return WindVector((x, y), velocity=(vx, vy))

    def __get_vector__(self, x: float, y: float, *args, **kwargs) -> WindVector:
        return self._interpolate(x, y, *args, **kwargs)
    
    def plot(self, lim: tuple[tuple, tuple], nx=30, ny=30, *args, **kwargs):
        self.__plot__(lim, nx, ny, *args, **kwargs)
        for w in self._collection:
            plt.quiver(w.x, w.y, w.vx, w.vy, color='red', scale=80)
    
    @property
    def x_max(self) -> float:
        return self._collection.x_max
    
    @property
    def x_min(self) -> float:
        return self._collection.x_min

    @property
    def y_max(self) -> float:
        return self._collection.y_max
    
    @property
    def y_min(self) -> float:
        return self._collection.y_min
    
def test():
    """
    Test function for MeasuredWindSource class.
    """
    p1, v1 = (1.0, 0.0), (-5., -2.)
    w1 = WindVector(p1, vector=v1)

    p2, v2 = (5.0, 10.), (-2., 3.)
    w2 = WindVector(p2, vector=v2)
    
    p3, v3 = (-5., -5.), (10., 5.)
    w3 = WindVector(p3, vector=v3)

    p4, v4 = (-2., 6.), (10., 5.)
    w4 = WindVector(p4, vector=v4)

    p5, v5 = (2., -4.), (10., 5.)
    w5 = WindVector(p5, vector=v5)

    src = MeasuredWindSource([w1, w2, w3, w4, w5])
    lim = ((src.x_min, src.y_min), (src.x_max, src.y_max))
    src.plot(lim, nx=20, ny=20, method='cubic')
    plt.show()

if __name__ == "__main__":
    test()