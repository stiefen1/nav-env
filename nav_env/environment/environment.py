from nav_env.ships.collection import ShipCollection
from nav_env.obstacles.collection import ObstacleWithKinematicsCollection, ObstacleCollection
from nav_env.geometry.vector_source import VectorSource
from nav_env.water.water_source import WaterSource
from nav_env.wind.wind_source import WindSource
import matplotlib.pyplot as plt
from nav_env.control.command import GeneralizedForces


class NavigationEnvironment:
    def __init__(self,
                 own_ships:list = None,
                 target_ships:list = None,
                 obstacles: list = None,
                 shore: list = None,
                 wind_source:WindSource = None,
                 water_source:WaterSource = None
                 ): 
        self._own_ships = ShipCollection(own_ships or [])
        self._target_ships = ShipCollection(target_ships or [])
        self._obstacles = ObstacleWithKinematicsCollection(obstacles or []) # TODO: Separate shore from moving obstacles as we might want to consider them separately, e.g. for TTG
        self._shore = ObstacleCollection(shore or [])
        self._wind_source = wind_source or WindSource()
        self._water_source = water_source or WaterSource()

    def step(self, external_forces:GeneralizedForces=GeneralizedForces()):
        """
        Step the environment.
        """
        # a ship also contains a controller
        # the environment only applies external conditions such as wind, water, obstacles
        self._own_ships.step(self._wind_source, self._water_source, external_forces=external_forces)
        self._target_ships.step(self._wind_source, self._water_source, external_forces=external_forces)

    def plot(self, t:float, lim:tuple, ax=None, own_ship_physics=['enveloppe', 'frame', 'acceleration', 'velocity', 'forces'], target_ship_physics=['enveloppe'], **kwargs):
        """
        Plot the environment.
        """
        if ax is None:
            _, ax = plt.subplots()
        self._shore.plot(ax=ax, **kwargs)
        self._own_ships.plot(ax=ax, keys=own_ship_physics, **kwargs)
        self._target_ships.plot(ax=ax, keys=target_ship_physics, **kwargs)
        self._obstacles(t).plot(ax=ax, **kwargs)
        self._wind_source.quiver(lim, ax=ax, facecolor='grey', alpha=0.1, **kwargs)
        # self._water_source.plot(ax=ax, **kwargs)
        ax.set_xlim((lim[0][0], lim[1][0]))
        ax.set_ylim((lim[0][1], lim[1][1]))
        return ax
    
    def draw(self, t:float, screen, own_ship_physics=['enveloppe', 'frame', 'acceleration', 'velocity', 'forces'], target_ship_physics=['enveloppe'], scale=1, **kwargs):
        """
        Draw the environment for pygame.
        """
        self._shore.draw(screen, scale=scale, **kwargs)
        self._obstacles(t).draw(screen, scale=scale, **kwargs)
        self._own_ships.draw(screen, keys=own_ship_physics, scale=scale, **kwargs)
        self._target_ships.draw(screen, keys=target_ship_physics, scale=scale, **kwargs)
        # self._wind_source.draw(screen, **kwargs)
        # self._water_source.draw(screen, **kwargs)
    
    def __repr__(self):
        return f"NavigationEnvironment({len(self._obstacles)} obstacles, {self._wind_source})"

    def to_dict(self) -> dict:
        return {
            'own_ships': self._own_ships,
            'target_ships': self._target_ships,
            'obstacles': self._obstacles,
            'shore': self._shore,
            'wind_source': self._wind_source,
            'water_source': self._water_source
        }
    
    def from_dict(self, d:dict) -> None:
        self._own_ships = d['own_ships']
        self._target_ships = d['target_ships']
        self._obstacles = d['obstacles']
        self._shore = d['shore']
        self._wind_source = d['wind_source']
        self._water_source = d['water_source']
    
    @property
    def shore(self) -> ObstacleCollection:
        return self._shore

    @property
    def obstacles(self) -> ObstacleWithKinematicsCollection:
        return self._obstacles
    
    @property
    def wind_source(self) -> WindSource:
        return self._wind_source
    
    @property
    def water_source(self) -> WaterSource:
        return self._water_source

    @property
    def own_ships(self) -> ShipCollection:
        return self._own_ships
    
    @property
    def target_ships(self) -> ShipCollection:
        return self._target_ships
    

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