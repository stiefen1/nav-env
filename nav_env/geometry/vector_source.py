from abc import ABC, abstractmethod
from shapely import Polygon
from nav_env.geometry.utils import *
from nav_env.geometry.vector import Vector
import matplotlib.pyplot as plt

LOC_HASHMAP = {
    "upper right": (1, 1),
    "upper left": (-1, 1),
    "lower right": (1, -1),
    "lower left": (-1, -1),
    "center": (0, 0),
    "center right": (1, 0),
    "center left": (-1, 0),
    "lower center": (0, -1),
    "upper center": (0, 1)
}

LABEL_HASHMAP = {
    'SI': (1.0, 'm/s'),
    'kn': (1.94384001, 'kn')
}

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
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        for xi in x:
            for yi in y:
                vec = self((xi, yi))
                ax.quiver(vec.x+dx/2, vec.y+dy/2, vec.vx, vec.vy, label=label, **kwargs)
                label = None
        return ax
    
    def compass(self, lim:tuple[tuple, tuple], *args, ax=None, label:str=None, size:float=0.2, loc:str='upper right', alpha:float=0.8, **kwargs):
        """
        size = compass diameter / smallest side
        """
        assert size <= 1 and size >= 0.01, f"size must be in range [0.01, 1] but is {size}"
        if ax is None:
            _, ax = plt.subplots()

        x_min, y_min = lim[0]
        x_max, y_max = lim[1]
        center = (x_min+x_max)/2, (y_min+y_max)/2

        # Diameter
        dy, dx = abs(y_max - y_min), (x_max - x_min)
        dmin = min(dy, dx)
        D = dmin * size # Compass' diameter

        # center
        d = D/8 # distance between outter circle and limits
        loc_dir = LOC_HASHMAP[loc]
        x, y = (dx/2-d-D/2)*loc_dir[0]+center[0], (dy/2-d-D/2)*loc_dir[1]+center[1]
        outter_circle = plt.Circle((x, y), radius=D/2, facecolor='white', edgecolor='black', alpha=alpha)
        vec = self((x, y)).normalize() * D * 0.75
        
        if label is not None:
            assert label in LABEL_HASHMAP.keys(), f"label can take values {LABEL_HASHMAP.keys()} but is {label}"
            dist_text_to_center = 1/3.2
            ax.text(x-vec.vy*dist_text_to_center, y+vec.vx*dist_text_to_center, f'{self((x, y)).norm()*LABEL_HASHMAP[label][0]:.1f}{LABEL_HASHMAP[label][1]}', color='black', size=8, ha='center')
            ax.text(x+vec.vy*dist_text_to_center, y-vec.vx*dist_text_to_center, 'Wind', color='black', size=8, ha='center')

        ax.add_patch(outter_circle)
        ax.quiver(vec.x-vec.vx/2, vec.y-vec.vy/2, vec.vx, vec.vy, scale_units='xy', scale=1, facecolor='black', width=0.003)
        

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
    
