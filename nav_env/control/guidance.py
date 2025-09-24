from math import atan2, pi, sqrt
import numpy as np
from nav_env.control.path import Waypoints
from abc import ABC, abstractmethod
from nav_env.ships.states import States3
from nav_env.colav.colav import COLAVBase, COLAV
from nav_env.ships.moving_ship import MovingShip

class GuidanceBase(ABC):
    def __init__(
            self,
            waypoints: Waypoints,
            current_wpt_idx:int,
            radius_of_acceptance:float,
            *args,
            colav:COLAVBase=None,
            **kwargs
        ):
        self._waypoints = waypoints
        self._current_wpt_idx = current_wpt_idx # Current waypoint is the one we just reached
        self._radius_of_acceptance = radius_of_acceptance
        self.colav = colav or COLAV(0)

    # def init_colav(self, colav:COLAVBase) -> None:
    #     self._colav = COLAV() if colav is None else colav
    #     self._colav.guidance = self

    def get(self, state:States3, *args, target_ships:list[MovingShip]=[], **kwargs) -> tuple[States3, dict]:
        commanded_state, info = self.__get__(state, *args, **kwargs)
        colav_des_state = self.colav.get(state, commanded_state, target_ships, *args, **kwargs)
        # print(colav_des_state, commanded_state)
        return colav_des_state, info

    @abstractmethod
    def __get__(self, state:States3, *args, **kwargs) -> tuple[States3, dict]:
        pass

    def reset(self) -> None:
        pass

    def get_next_waypoint(self):
        return self._waypoints[self._current_wpt_idx+1]
    
    def get_prev_waypoint(self):
        if self._current_wpt_idx > 0:
            return self._waypoints[self._current_wpt_idx-1]
        else:
            print("Unable to fetch a previous waypoint, the initial waypoint will be returned instead..")
            return self._waypoints[0]
    
    def next_waypoint(self):
        # If current waypoint is the final waypoint do not increment
        if self._current_wpt_idx >= self.n - 1:
            return
        self._current_wpt_idx += 1
        return
    
    def within_radius_of_acceptance(self, x, y) -> bool:
        wx, wy = self.current_waypoint
        d2 = ((wx-x)**2 + (wy-y)**2)
        # print(wx, wy, x, y)
        # print(d2, self._radius_of_acceptance**2, abs(wx-x), abs(wy-y))
        return d2 <= self._radius_of_acceptance**2
    
    def distance_to_final_waypoint(self, x, y) -> bool:
        wf = self._waypoints[-1]
        return ((wf[0]-x)**2 + (wf[1]-y)**2)**0.5
    
    @property
    def n(self) -> int:
        return len(self._waypoints)
    
    @property
    def current_waypoint(self) -> tuple[float, float]:
        return self._waypoints[self._current_wpt_idx]
    
    @property
    def current_idx(self) -> int:
        return self._current_wpt_idx
    
class Guidance(GuidanceBase):
    """Just a default class to be instantied if nothing else is provided --> guidance = guidance or Guidance()"""
    def __init__(self, waypoints: Waypoints=[], current_wpt_idx:int=0, radius_of_acceptance:float=50., *args, **kwargs):
        super().__init__(waypoints=waypoints, current_wpt_idx=current_wpt_idx, radius_of_acceptance=radius_of_acceptance, *args, **kwargs)

    def __get__(self, state:States3, *args, **kwargs) -> tuple[States3, dict]:
        return state, {}
    
class PathProgressionAndSpeedGuidance(GuidanceBase):
    """
    Very simple guidance system to store waypoints and reference speed. Useful for control algorithms utilizing path progression.
    """
    def __init__(self, waypoints: Waypoints, speed_ref: float, *args, **kwargs):
        self._speed_ref = speed_ref
        super().__init__(waypoints=waypoints, current_wpt_idx=0, radius_of_acceptance=50, *args, **kwargs)

    def __get__(self, *args, **kwargs) -> tuple[States3, dict]:
        """
        Returns desired speed to be tracked
        """
        return States3(x_dot=self._speed_ref), {}
    
class TrajectoryTrackingGuidance(GuidanceBase):
    pass

class ConstantHeadingAndSpeed(GuidanceBase):
    def __init__(self, desired_heading_deg:float, desired_speed:float, *args, **kwargs):
        super().__init__(None, None, None, *args, **kwargs)
        self.desired_heading_deg = desired_heading_deg
        self.desired_speed = desired_speed

    def __get__(self, *args, **kwargs) -> tuple[States3, dict]:
        return States3(psi_deg=self.desired_heading_deg, x_dot=self.desired_speed), {}

    