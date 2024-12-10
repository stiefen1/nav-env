"""
Ideal usage:
s1 = ShipObstacle(length=l, width=w, p0=p0, v0=v0)
s2 = ShipObstacle(domain=domain, p_t=lambda t: p(t))

"""

from nav_env.obstacles.obstacles import ObstacleWithKinematics
from nav_env.obstacles.obstacles import Obstacle
from typing import Callable
from math import atan2, pi
from copy import deepcopy
import os, pathlib, sys
from nav_env.ships.states import States3, States2

PATH_TO_DEFAULT_IMG = os.path.join(pathlib.Path(__file__).parent.parent, "ships", "ship.png")

DEFAULT_TARGET_SHIP_LENGTH = 10.
DEFAULT_TARGET_SHIP_WIDTH = 4.
DEFAULT_TARGET_SHIP_RATIO = 10/15 # = length of the lower rectangle / tot length
DEFAULT_TARGET_SHIP_SPEED = (1., 1., 0.)
DEFAULT_TARGET_SHIP_POSITION = (0., 0., 0.)

def get_target_ship_domain(length, width, ratio):
    # Make the centroid of this domain placed at (0, 0)

    # Upper triangle area:
    triangle_height = (1-ratio) * length
    triangle_area = width * triangle_height / 2
    # Upper triange centroid y:
    triangle_centroid_y = triangle_height / 3
    # Lower rectangle area:
    rectangle_height = ratio * length
    rectangle_area = rectangle_height * width
    # Lower rectangle centroid y:
    rectangle_centroid_y = -rectangle_height / 2
    # Overall centroid
    centroid_y = (triangle_area * triangle_centroid_y + rectangle_area * rectangle_centroid_y) / (triangle_area + rectangle_area)

    domain = [
        [0, triangle_height - centroid_y],
        [-width/2, 0 - centroid_y],
        [-width/2, -rectangle_height-centroid_y],
        [width/2, -rectangle_height-centroid_y],
        [width/2, 0 - centroid_y]
    ]

    return domain

class ShipEnveloppe(Obstacle):
    def __init__(self,
                 xy: list=None,
                 length: float=DEFAULT_TARGET_SHIP_LENGTH,
                 width: float=DEFAULT_TARGET_SHIP_WIDTH,
                 ratio: float=DEFAULT_TARGET_SHIP_RATIO,
                 img:str=PATH_TO_DEFAULT_IMG
                 ):
        
        if xy is None:
            xy = get_target_ship_domain(length, width, ratio)

        super().__init__(xy=xy, img=img)

    def plot(self, ax=None, c='r', alpha=1, **kwargs):
        return super().plot(ax=ax, c=c, alpha=alpha, **kwargs)

class ShipWithKinematics(ObstacleWithKinematics):
    """
    A target ship that moves according to either:
    - a given pose function p_t: t -> (x, y, heading)
    - a given initial position p0 and speed v0
    The flag make_heading_consistent allows to make the heading consistent with the trajectory, i.e. aligned with the velocity vector.
    """
    def __init__(self,
                 length: float=DEFAULT_TARGET_SHIP_LENGTH,
                 width: float=DEFAULT_TARGET_SHIP_WIDTH,
                 ratio: float=DEFAULT_TARGET_SHIP_RATIO,
                 pose_fn: Callable[[float], States3]=None,
                 initial_states: States2=None,
                 id:int=None,
                 **kwargs
                 ):
            
        enveloppe = ShipEnveloppe(length=length, width=width, ratio=ratio, **kwargs)
        initial_states = initial_states if initial_states is None else States3(*initial_states.xy, 0, *initial_states.xy_dot, 0)
        super().__init__(pose_fn=pose_fn, initial_state=initial_states, xy=enveloppe.get_xy_as_list(), id=id)

    def get_pose_at(self, t:float) -> States3:
        """
        Override get_pose_at to make the heading consistent with the trajectory.
        """
        dt = 1e-2 # small time step to compute the numerical derivative
        s0 = self._initial_state
        x1, x2 = s0.x + self._initial_state.x_dot * t, s0.x + s0.x_dot * (t+dt)
        y1, y2 = s0.y + s0.y_dot * t, s0.y + s0.y_dot * (t+dt)
        dxdt = (x2 - x1)/dt
        dydt = (y2 - y1)/dt
        heading = atan2(dydt, dxdt) * 180 / pi - 90
        return States3(x1, y1, heading, s0.x_dot, s0.y_dot, 0)        

def test():
    import matplotlib.pyplot as plt
    from nav_env.obstacles.collection import ObstacleWithKinematicsCollection
    from nav_env.ships.states import States2
    import numpy as np
    from math import cos, sin


    p = lambda t: States3(x=-10*cos(0.2*t), y=8*sin(0.2*t))
    Ts1 = ShipWithKinematics(pose_fn=p)
    Ts2 = ShipWithKinematics(length=30, width=10, ratio=7/9, initial_states=States2(0, 0, 1, 1))
    Ts3 = ShipWithKinematics(width=8, ratio=3/7, pose_fn=lambda t: States3(t, -t, t*10))
    coll = ObstacleWithKinematicsCollection([Ts1, Ts2, Ts3])
    
    fig2 = plt.figure(2)
    ax = fig2.add_axes(111, projection='3d')
    for t in np.linspace(0, 32, 32):
        coll.plot3(t, ax=ax)
    plt.show()

if __name__ == "__main__":
    test()