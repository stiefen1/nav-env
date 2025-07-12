from abc import ABC, abstractmethod
from typing import Any
from nav_env.ships.states import States3
import numpy as np
from nav_env.sensors.collection import SensorCollection

class NavigationBase(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def observe(self, *args, **kwargs) -> Any:
        pass

    def reset(self) -> None:
        pass

class Navigation(NavigationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def observe(self, ship, *args, **kwargs) -> States3:
        sensors:SensorCollection = ship.sensors
        measurements = sensors.get()
        R3d = ship._physics.rotation_matrix(ship.states.psi_rad, dim=3)
        pose_dot_in_ship_frame = np.dot(R3d, ship.states.vel)
        return States3(
            x=ship.states.x,
            y=ship.states.y,
            psi_deg=ship.states.psi_deg,
            x_dot=pose_dot_in_ship_frame[0],
            y_dot=pose_dot_in_ship_frame[1],
            psi_dot_deg=pose_dot_in_ship_frame[2]*180/np.pi)
    
def test() -> None:
    from nav_env.ships.ship import Ship
    nav = Navigation()
    ship = Ship()
    obs = nav.observe(ship)
    print(obs)

if __name__ == "__main__":
    test()