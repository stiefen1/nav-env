"""
Abstract base class and concrete implementations for wind sources, including methods for interpolating wind data and plotting wind fields.
"""

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from shapely import Point, Polygon
from .wind_vector import WindVector, WindVectorCollection
from .utils import *
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

class WindSource(ABC):
    """
    Abstract base class for wind sources.
    """
    def __init__(self, domain: Polygon = None):
        self._domain = domain

    def __call__(self, position: np.ndarray | Point | tuple, *args, **kwargs) -> WindVector:
        """
        Get the wind vector at a given position.
        """
        position: tuple[float, float] = convert_any_to_tuple(position)
        assert_tuple_2d_position(position)
        return self.__get_wind__(*position, *args, **kwargs)
    
    def __plot__(self, lim: tuple[tuple, tuple], nx=30, ny=30, *args, **kwargs):
        """
        Plot the wind field over a specified range.
        """
        x_min, y_min = lim[0]
        x_max, y_max = lim[1]
        x_mean, y_mean = (x_min + x_max) / 2, (y_min + y_max) / 2
        
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        for i in range(len(x)):
            for j in range(len(y)):
                try:
                    zz[j, i] = self((x[i], y[j]), *args, **kwargs).intensity
                except:
                    zz[j, i] = np.nan
        _, ax = plt.subplots()
        cont = ax.contourf(xx, yy, zz)
        plt.colorbar(cont)
        ax.set_xlim((x_min - x_mean) * 1.2 + x_mean, (x_max - x_mean) * 1.2 + x_mean)
        ax.set_ylim((y_min - y_mean) * 1.2 + y_mean, (y_max - y_mean) * 1.2 + y_mean)

    @abstractmethod
    def __get_wind__(self, x: float, y: float, *args, **kwargs) -> WindVector:
        """
        Abstract method to get the wind vector at a given position.
        """
        pass

    @abstractmethod
    def plot(self, lim: tuple[tuple, tuple], nx=30, ny=30, *args, **kwargs):
        """
        Abstract method to plot the wind field.
        """
        pass

    @property
    def domain(self) -> Polygon:
        return self._domain

class CallableWindSource(WindSource):
    """
    Wind source defined by a callable function.
    """
    def __init__(self, wind_function: Callable[[tuple[float, float]], WindVector], domain: Polygon = None):
        super().__init__(domain)
        self._wind_function = wind_function

    def __get_wind__(self, x: float, y: float, *args, **kwargs) -> WindVector:
        return self._wind_function(x, y, *args, **kwargs)
    
    def plot(self, lim: tuple[tuple, tuple], nx=30, ny=30, *args, **kwargs):
        self.__plot__(lim, nx, ny, *args, **kwargs)

class MeasuredWindSource(WindSource):
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

    def __get_wind__(self, x: float, y: float, *args, **kwargs) -> WindVector:
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
    w1 = WindVector(p1, velocity=v1)

    p2, v2 = (5.0, 10.), (-2., 3.)
    w2 = WindVector(p2, velocity=v2)
    
    p3, v3 = (-5., -5.), (10., 5.)
    w3 = WindVector(p3, velocity=v3)

    p4, v4 = (-2., 6.), (10., 5.)
    w4 = WindVector(p4, velocity=v4)

    p5, v5 = (2., -4.), (10., 5.)
    w5 = WindVector(p5, velocity=v5)

    src = MeasuredWindSource([w1, w2, w3, w4, w5])
    lim = ((src.x_min, src.y_min), (src.x_max, src.y_max))
    src.plot(lim, nx=20, ny=20, method='cubic')
    plt.show()

if __name__ == "__main__":
    test()