from nav_env.obstacles.collection import ObstacleCollection
from nav_env.geometry.vector_source import VectorSource
from nav_env.water.water_source import WaterSource
from nav_env.wind.wind_source import MeasuredWindSource
import matplotlib.pyplot as plt
# from nav_env.ships.collection import ShipCollection

class NavigationEnvironment:
    def __init__(self,
                 obstacles: ObstacleCollection = ObstacleCollection(),
                 wind_source:VectorSource = MeasuredWindSource(),
                 water_source:WaterSource = WaterSource()
                #  ships: ShipCollection = ShipCollection()
                 ): 
        self._obstacles = obstacles
        self._wind_source = wind_source
        self._water_source = water_source
        # self._ships = ships

    def plot(self, ax=None, **kwargs):
        """
        Plot the environment.
        """
        if ax is None:
            _, ax = plt.subplots()
        self._obstacles.plot(ax=ax, **kwargs)
        self._wind_source.plot(ax=ax, **kwargs)
        return ax
    
    def draw(self):
        """
        Draw the environment for pygame.
        """
        self._obstacles.draw()
        self._wind_source.draw()
        self._water_source.draw()
        # self._ships.draw()
    
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