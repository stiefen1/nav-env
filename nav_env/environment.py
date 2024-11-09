from nav_env.obstacles.collection import ObstacleCollection
from nav_env.geometry.vector_source import VectorSource
from nav_env.water.water_source import WaterSource
import matplotlib.pyplot as plt
class NavigationEnvironment:
    def __init__(self, obstacles: ObstacleCollection = ObstacleCollection(), wind_source:VectorSource = VectorSource(), water_source:WaterSource = WaterSource()): 
        self._obstacles = obstacles
        self._wind_source = wind_source
        self._water_source = water_source

    def plot(self, ax=None, **kwargs):
        """
        Plot the environment.
        """
        if ax is None:
            _, ax = plt.subplots()
        self._obstacles.plot(ax=ax, **kwargs)
        self._wind_source.plot(ax=ax, **kwargs)
        return ax
    
    def __repr__(self):
        return f"NavigationEnvironment({len(self._obstacles)} obstacles, {self._wind_source})"
    
    @property
    def obstacles(self) -> ObstacleCollection:
        return self._obstacles
    
    @property
    def wind_source(self) -> VectorSource:
        return self._wind_source
    
    @property
    def water_source(self) -> WaterSource:
        return self._water_source