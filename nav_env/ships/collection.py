from nav_env.ships.ship import SimpleShip, ShipWithDynamicsBase
from nav_env.ships.moving_ship import MovingShip
import matplotlib.pyplot as plt
# from nav_env.environment.disturbances import DisturbanceCollection
from nav_env.control.command import GeneralizedForces
from nav_env.wind.wind_source import WindSource
from nav_env.water.water_source import WaterSource

class ShipCollection:
    def __init__(self, ships: list[MovingShip] = None):
        assert isinstance(ships, list), f"Expected list got {type(ships).__name__}"
        self._ships = ships or []

    def step(self, wind:WindSource, water:WaterSource, external_forces:GeneralizedForces=GeneralizedForces()):
        """
        Step all ships.
        """
        for ship in self._ships:
            xy = ship.states.xy
            # print(ship.name, xy)
            ship.step(wind(xy), water(xy), external_forces=external_forces)

    def reset(self):
        """
        Reset all ships.
        """
        for ship in self._ships:
            ship.reset()

    def plot(self, ax=None, params:dict={'enveloppe':1}, **kwargs):
        """
        Plot the ships.
        """
        if ax is None:
            _, ax = plt.subplots()

        for ship in self._ships:
            ship.plot(ax=ax, params=params, **kwargs)

        return ax
    
    def draw(self, screen, *args, params:dict={'enveloppe':1}, scale=1, **kwargs):
        """
        Draw the ships.
        """
        for ship in self._ships:
            ship.draw(screen, params=params, scale=scale, **kwargs)

    def get_except(self, ship: MovingShip):
        """
        Get all ships except the given ship.
        """
        return ShipCollection([s for s in self._ships if s != ship])
    
    def set_integration_step(self, dt: float):
        """
        Set the integration step for all ships.
        """
        for ship in self._ships:
            ship.dt = dt

    def __setitem__(self, index: int, ship: MovingShip):
        self._ships[index] = ship

    def __getitem__(self, index: int) -> MovingShip:
        return self._ships[index]
    
    def append(self, ship):
        if isinstance(ship, MovingShip):
            self._ships.append(ship)
        elif isinstance(ship, ShipCollection):
            self._ships.extend(ship._ships)
        else:
            raise ValueError(f"Expected ShipWithDynamicsBase or ShipCollection got {type(ship)}")

    def remove(self, ship: MovingShip):
        self._ships.remove(ship)

    def __len__(self):
        return len(self._ships)
    
    def __repr__(self):
        return f"{type(self).__name__}({len(self._ships)} ships)"
    
    def __iter__(self):
        for ship in self._ships:
            yield ship


def test():
    
    # TODO: Fix issue with __dict__when printing ships[0]
    ship1 = SimpleShip()
    ship2 = SimpleShip()
    ship3 = SimpleShip()
    ships = ShipCollection([ship1, ship2])
    new_coll = ships.get_except(ship2)
    ships.append(ship3)
    print(new_coll, ships[0])
    assert len(ships) == 3, f"Expected 3 ship got {len(ships)}"
    assert ships[0] == ship1, f"Expected {ship1} got {ships[0]}"
    ships.remove(ship1)
    assert len(ships) == 2, f"Expected 2 ships got {len(ships)}"
    # new_coll = ships.get_except(ship2)
    assert len(new_coll) == 1, f"Expected 1 ship got {len(new_coll)}"    
    print("PASSED")

if __name__ == "__main__":
    test()