from abc import ABC, abstractmethod
from nav_env.ships.states import States3
from nav_env.ships.moving_ship import MovingShip

class COLAVBase(ABC):
    def __init__(self, distance_threshold:float, *args, distance_margin:float=0.0, **kwargs) -> None:
        self.distance_threshold = distance_threshold       # If an obstacle is at a distance < distance_threshold, we switch to colav mode
        self.distance_margin = distance_margin

    def get(self, state:States3, commanded_state:States3, target_ships:list[MovingShip], *args, **kwargs) -> States3:
        return self.__get__(state, commanded_state, target_ships, *args, **kwargs)
    
    @abstractmethod
    def __get__(self, state:States3, commanded_state:States3, target_ships:list[MovingShip], *args, **kwargs) -> States3:
        return commanded_state

class COLAV(COLAVBase):
    def __init__(self, distance_threshold:float, *args, distance_margin:float=0.0, **kwargs) -> None:
        super().__init__(distance_threshold, *args, distance_margin=distance_margin, **kwargs)

    def __get__(self, state:States3, commanded_state:States3, target_ships:list[MovingShip], *args, **kwargs) -> States3:
        return super().__get__(state, commanded_state, target_ships, *args, **kwargs)