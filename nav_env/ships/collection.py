from nav_env.ships.ship import SimpleShip, ShipWithDynamicsBase
from copy import deepcopy

class ShipCollection:
    def __init__(self, ships: list[ShipWithDynamicsBase] = []):
        self._ships = ships

    def plot(self, ax=None, **kwargs):
        """
        Plot the ships.
        """
        for ship in self._ships:
            ship.plot(ax=ax, **kwargs)
        return ax
    
    def draw(self, screen):
        """
        Draw the ships.
        """
        for ship in self._ships:
            ship.draw(screen)

    def get_except(self, ship: ShipWithDynamicsBase):
        """
        Get all ships except the given ship.
        """
        return ShipCollection([s for s in self._ships if s != ship])

    def __setitem__(self, index: int, ship: ShipWithDynamicsBase):
        self._ships[index] = ship

    def __getitem__(self, index: int) -> ShipWithDynamicsBase:
        return self._ships[index]
    
    def append(self, ship):
        if isinstance(ship, ShipWithDynamicsBase):
            self._ships.append(ship)
        elif isinstance(ship, ShipCollection):
            self._ships.extend(ship._ships)
        else:
            raise ValueError(f"Expected ShipWithDynamicsBase or ShipCollection got {type(ship)}")

    def remove(self, ship: ShipWithDynamicsBase):
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
    from nav_env.ships.states import ShipStates3
    ship1 = SimpleShip(ShipStates3(), None)
    ship2 = SimpleShip(ShipStates3(), None)
    ship3 = SimpleShip(ShipStates3(), None)
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