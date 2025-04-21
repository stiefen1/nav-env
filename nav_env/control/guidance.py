from math import atan2, pi, sqrt
import numpy as np
from nav_env.control.path import Waypoints
from abc import ABC, abstractmethod

class Guidance(ABC):
    def __init__(self, waypoints: Waypoints, current_wpt_idx:int, radius_of_acceptance:float, *args, **kwargs):
        self._waypoints = waypoints
        self._current_wpt_idx = current_wpt_idx # Current waypoint is the one we just reached
        self._radius_of_acceptance = radius_of_acceptance

    @abstractmethod
    def get_desired_heading(self, x, y, *args, degree=False, **kwargs) -> float:
        pass

    def get_next_waypoint(self):
        return self._waypoints[self._current_wpt_idx+1]
    
    def next_waypoint(self):
        # If we reached the final waypoint do not increment
        if self._current_wpt_idx >= self.n - 1:
            return
        self._current_wpt_idx += 1
        return
    
    def within_radius_of_acceptance(self, x, y) -> bool:
        wx, wy = self.get_next_waypoint()
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