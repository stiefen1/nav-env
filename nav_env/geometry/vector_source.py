from abc import ABC, abstractmethod
from shapely import Polygon
from nav_env.geometry.utils import *
from nav_env.geometry.vector import Vector
import matplotlib.pyplot as plt

class VectorSource(ABC):
    """
    Abstract base class for wind sources.
    """
    def __init__(self, domain: Polygon = None):
        self._domain = domain

    def __call__(self, position: np.ndarray | Point | tuple, *args, **kwargs) -> Vector:
        """
        Get the wind vector at a given position.
        """
        position: tuple[float, float] = convert_any_to_tuple(position)
        assert_tuple_2d_position(position)
        return self.__get_vector__(*position, *args, **kwargs)
    
    def __plot__(self, lim: tuple[tuple, tuple], *args, nx=10, ny=10, ax=None, **kwargs):
        """
        Plot the wind field over a specified range.
        """
        if ax is None:
            _, ax = plt.subplots()

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
        cont = ax.contourf(xx, yy, zz)
        # plt.colorbar(cont, ax=ax)
        ax.set_xlim((x_min - x_mean) * 1.2 + x_mean, (x_max - x_mean) * 1.2 + x_mean)
        ax.set_ylim((y_min - y_mean) * 1.2 + y_mean, (y_max - y_mean) * 1.2 + y_mean)
        return ax
    
    def quiver(self, lim: tuple[tuple, tuple], *args, nx=3, ny=3, ax=None, **kwargs):
        """
        Plot the wind field over a specified range.
        """
        if ax is None:
            _, ax = plt.subplots()

        x_min, y_min = lim[0]
        x_max, y_max = lim[1]

        label = None
        if 'label' in kwargs.keys():
            label = kwargs['label']
            kwargs.pop('label')
        
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        for xi in x:
            for yi in y:
                vec = self((xi, yi))
                ax.quiver(vec.x, vec.y, vec.vx, vec.vy, label=label, **kwargs)
                label = None
        return ax

    def draw():
        """
        Draw the vector source for pygame.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def __get_vector__(self, x: float, y: float, *args, **kwargs) -> Vector:
        """
        Abstract method to get the vector at a given position.
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