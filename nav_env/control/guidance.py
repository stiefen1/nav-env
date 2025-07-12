from math import atan2, pi, sqrt
import numpy as np
from nav_env.control.path import Waypoints
from abc import ABC, abstractmethod
from nav_env.ships.states import States3

class GuidanceBase(ABC):
    def __init__(self, waypoints: Waypoints, current_wpt_idx:int, radius_of_acceptance:float, *args, **kwargs):
        self._waypoints = waypoints
        self._current_wpt_idx = current_wpt_idx # Current waypoint is the one we just reached
        self._radius_of_acceptance = radius_of_acceptance

    @abstractmethod
    def get(self, state:States3, *args, **kwargs) -> tuple[States3, dict]:
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
        # If we reached the final waypoint do not increment
        if self._current_wpt_idx >= self.n - 1:
            return
        self._current_wpt_idx += 1
        return
    
    def within_radius_of_acceptance(self, x, y) -> bool:
        wx, wy = self.current_waypoint
        return ((wx-x)**2 + (wy-y)**2) <= self._radius_of_acceptance**2
    

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

    def get(self, state:States3, *args, **kwargs) -> tuple[States3, dict]:
        return state, {}
    
class PathProgressionAndSpeedGuidance(GuidanceBase):
    """
    Very simple guidance system to store waypoints and reference speed. Useful for control algorithms utilizing path progression.
    """
    def __init__(self, waypoints: Waypoints, speed_ref: float, *args, **kwargs):
        self._speed_ref = speed_ref
        super().__init__(waypoints=waypoints, current_wpt_idx=0, radius_of_acceptance=50, *args, **kwargs)

    def get(self, *args, **kwargs) -> tuple[States3, dict]:
        """
        Returns desired speed to be tracked
        """
        return States3(x_dot=self._speed_ref), {}
    
class TrajectoryTrackingGuidance(GuidanceBase):
    pass

    