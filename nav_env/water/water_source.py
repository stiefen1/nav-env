from nav_env.geometry.vector_source import VectorSource
from nav_env.geometry.vector import Vector

class WaterSource(VectorSource):
    def __init__(self):
        pass

    def __get_vector__(self, x: float, y: float, *args, **kwargs) -> Vector:
        """
        Abstract method to get the vector at a given position.
        """
        pass

    def plot(self, lim: tuple[tuple, tuple], nx=30, ny=30, *args, **kwargs):
        """
        Plot the wind field over a specified range.
        """
        self.__plot__(lim, nx, ny, *args, **kwargs)
        