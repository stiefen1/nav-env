from nav_env.geometry.vector_source import VectorSource
from nav_env.geometry.vector import Vector
from nav_env.water.water_vector import WaterVector
from typing import Callable
from shapely import Polygon


class WaterSource(VectorSource):
    def __init__(self, water_function: Callable[[tuple[float, float]], WaterVector] = None, domain: Polygon = None):
        super().__init__(domain)
        self._water_function = water_function or self.default_water_function

    def __get_vector__(self, x: float, y: float, *args, **kwargs) -> WaterVector:
        """
        Abstract method to get the vector at a given position.
        """
        return self._water_function(x, y, *args, **kwargs)

    def plot(self, lim: tuple[tuple, tuple], nx=30, ny=30, *args, **kwargs):
        """
        Plot the wind field over a specified range.
        """
        self.__plot__(lim, nx, ny, *args, **kwargs)

    def default_water_function(self, x: float, y: float) -> WaterVector:
        """
        Default wind function that returns a zero vector.
        """
        return WaterVector((x, y), vector=(0., 0.))

class UniformWaterSource(WaterSource):
    """
    Water source with a uniform water vector.
    """
    def __init__(self, velocity_x:float=0., velocity_y:float=0., domain: Polygon = None):
        self._velocity = (velocity_x, velocity_y)
        water_function = self.uniform_water_function
        super().__init__(water_function=water_function, domain=domain)
    
    def uniform_water_function(self, x, y):
        return WaterVector((x, y), vector=self._velocity)
        