from nav_env.ships.collection import ShipCollection
from nav_env.obstacles.collection import ObstacleWithKinematicsCollection, ObstacleCollection
from nav_env.geometry.vector_source import VectorSource
from nav_env.water.water_source import WaterSource
from nav_env.wind.wind_source import MeasuredWindSource
import matplotlib.pyplot as plt
from nav_env.environment.disturbances import DisturbanceCollection


class NavigationEnvironment:
    def __init__(self,
                 own_ships:ShipCollection = ShipCollection(),
                 target_ships:ShipCollection = ShipCollection(),
                 obstacles: ObstacleWithKinematicsCollection = ObstacleWithKinematicsCollection(),
                 shore: ObstacleCollection = ObstacleCollection(),
                 wind_source:VectorSource = MeasuredWindSource(),
                 water_source:WaterSource = WaterSource()
                 ): 
        self._own_ships, self._target_ships = own_ships, target_ships # We allow to have multiple own ships
        self._obstacles = obstacles # TODO: Separate shore from moving obstacles as we might want to consider them separately, e.g. for TTG
        self._shore = shore
        self._wind_source = wind_source
        self._water_source = water_source

    def step(self, external_forces:DisturbanceCollection=DisturbanceCollection()):
        """
        Step the environment.
        """
        # a ship also contains a controller
        # the environment only applies external conditions such as wind, water, obstacles
        for ship in self._own_ships:
            # wind = self._wind_source(ship.position)
            # water = self._water_source(ship.position)
            ship.step(DisturbanceCollection(), external_forces=external_forces)

    def plot(self, t:float, ax=None, **kwargs):
        """
        Plot the environment.
        """
        if ax is None:
            _, ax = plt.subplots()
        self._shore.plot(ax=ax, **kwargs)
        self._own_ships.plot(ax=ax, **kwargs)
        self._target_ships.plot(ax=ax, **kwargs)
        self._obstacles(t).plot(ax=ax, **kwargs)
        # self._wind_source.plot(ax=ax, **kwargs)
        # self._water_source.plot(ax=ax, **kwargs)
        return ax
    
    def draw(self, t:float, screen, *args, **kwargs):
        """
        Draw the environment for pygame.
        """
        self._shore.draw(screen, *args, **kwargs)
        self._obstacles(t).draw(screen, *args, **kwargs)
        self._wind_source.draw(screen, *args, **kwargs)
        self._water_source.draw(screen, *args, **kwargs)
    
    def __repr__(self):
        return f"NavigationEnvironment({len(self._obstacles)} obstacles, {self._wind_source})"
    
    @property
    def shore(self) -> ObstacleCollection:
        return self._shore

    @property
    def obstacles(self) -> ObstacleWithKinematicsCollection:
        return self._obstacles
    
    @property
    def wind_source(self) -> VectorSource:
        return self._wind_source
    
    @property
    def water_source(self) -> WaterSource:
        return self._water_source
    

def test():
    from nav_env.viz.matplotlib_screen import MatplotlibScreen as Screen
    from nav_env.ships.ship import SimpleShip, ShipTimeDerivatives3, ShipStates3
    from nav_env.simulation.integration import Euler

    os = SimpleShip(states=ShipStates3(1., 2., 10., 1., 1., 10.), derivatives=ShipTimeDerivatives3(1., 1., 10., 0., 0., 0.), integrator=Euler(0.03))
    ts = SimpleShip(states=ShipStates3(0., 0., 0., -1., 1., -10.), derivatives=ShipTimeDerivatives3(-1., 1., -10., 0., 0., 0.), integrator=Euler(0.03))
    ships = ShipCollection([os, ts])
    env = NavigationEnvironment(own_ships=ships)
    screen = Screen(env)
    print(os.integrator.dt)
    screen.play(dt=os.integrator.dt)

if __name__ == "__main__":
    test()