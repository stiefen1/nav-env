from nav_env.ships.ship import ShipBase

class ShipCollection:
    def __init__(self, ships: list[ShipBase] = []):
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

    def __setitem__(self, index: int, ship: ShipBase):
        self._ships[index] = ship

    def __getitem__(self, index: int) -> ShipBase:
        return self._ships[index]
    
    def append(self, ship: ShipBase):
        assert isinstance(ship, ShipBase), f"Ship must be an instance of Ship not {type(ship)}"
        self._ships.append(ship)

    def remove(self, ship: ShipBase):
        self._ships.remove(ship)

    def __len__(self):
        return len(self._ships)
    
    def __repr__(self):
        return f"SimpleShipCollection({len(self._ships)} ships)"
    
    def __iter__(self):
        for ship in self._ships:
            yield ship