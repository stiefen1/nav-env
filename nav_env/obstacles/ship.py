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
from abc import abstractmethod

DEFAULT_PATH_TO_IMG = os.path.join(pathlib.Path(__file__).parent.parent, "ships", "ship.png")

DEFAULT_TARGET_SHIP_LENGTH = 10.
DEFAULT_TARGET_SHIP_WIDTH = 4.
DEFAULT_TARGET_SHIP_RATIO = 10/15 # = length of the lower rectangle / tot length
DEFAULT_TARGET_SHIP_SPEED = (1., 1., 0.)
DEFAULT_TARGET_SHIP_POSITION = (0., 0., 0.)

def get_target_ship_domain(length, width, ratio, *args, **kwargs) -> list:
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

# TODO: Check why output of rotate_and_translate is not the same type
class EnveloppeBase(Obstacle):
    def __init__(self,
                 xy: list=None,
                 *args, **kwargs
                 ):
        if xy is None:
            xy = self.get_domain(*args, **kwargs)

        self._initial_domain = deepcopy(xy)
        super().__init__(xy=xy, *args, **kwargs)
    
    @abstractmethod
    def get_domain(self) -> list:
        pass

    def reset(self, *args, **kwargs):
        super().__init__(xy=self._initial_domain, *args, **kwargs)

    @property
    def initial_domain(self):
        return self._initial_domain
    

class EmptyEnveloppe(EnveloppeBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_domain(self) -> list:
        return []
    
class ShipEnveloppe(EnveloppeBase):
    def __init__(self,
                 xy: list=None,
                 length: float=None,
                 width: float=None,
                 ratio: float=None,
                 img:str=None
                 ):
        length = length or DEFAULT_TARGET_SHIP_LENGTH
        width = width or DEFAULT_TARGET_SHIP_WIDTH
        ratio = ratio or DEFAULT_TARGET_SHIP_RATIO
        img = img or DEFAULT_PATH_TO_IMG
        super().__init__(xy=xy, length=length, width=width, ratio=ratio, img=img)

    def get_domain(self, length, width, ratio, *args, **kwargs) -> list:
        return get_target_ship_domain(length, width, ratio, *args, **kwargs)
    


# class ShipWithKinematics(ObstacleWithKinematics):
#     """
#     A target ship that moves according to either:
#     - a given pose function p_t: t -> (x, y, heading)
#     - a given initial position p0 and speed v0
#     The flag make_heading_consistent allows to make the heading consistent with the trajectory, i.e. aligned with the velocity vector.
#     """
#     def __init__(self,
#                  enveloppe:ShipEnveloppe=None,
#                  pose_fn: Callable=None,
#                  p0: tuple[float, float, float]=DEFAULT_TARGET_SHIP_POSITION,
#                  v0: tuple[float, float, float]=DEFAULT_TARGET_SHIP_SPEED,
#                  make_heading_consistent:bool=False,
#                  **kwargs
#                  ):
        
#         if enveloppe is None:
#             enveloppe = ShipEnveloppe(**kwargs)

#         if pose_fn is None:
#             pose_fn = lambda t: (p0[0] + v0[0] * t, p0[1] + v0[1] * t, p0[2] + v0[2] * t)

#         if make_heading_consistent:
#             dt = 1e-2
#             new_pose_fn = deepcopy(pose_fn)
#             x = lambda t: new_pose_fn(t)[0]
#             y = lambda t: new_pose_fn(t)[1]
#             dxdt = lambda t : (x(t+dt) - x(t-dt))/(2*dt)
#             dydt = lambda t : (y(t+dt) - y(t-dt))/(2*dt)
#             heading = lambda t: atan2(dydt(t),(dxdt(t)))*180/pi - 90
#             pose_fn = lambda t: (x(t), y(t), heading(t))
        
#         super().__init__(pose_fn, xy=enveloppe.get_xy_as_list())


# def test_old():
#     import matplotlib.pyplot as plt
#     from nav_env.obstacles.collection import ObstacleWithKinematicsCollection
#     import numpy as np
#     from math import cos, sin


#     p = lambda t: (-10*cos(0.2*t), 8*sin(0.2*t), 0)
#     Ts1 = ShipWithKinematics(pose_fn=p, make_heading_consistent=True)
#     Ts2 = ShipWithKinematics(length=30, width=10, ratio=7/9, p0=(0, 0, 0), v0=(1, 1, 0), make_heading_consistent=True)
#     Ts3 = ShipWithKinematics(width=8, ratio=3/7, pose_fn=lambda t: (t, -t, t*10))
#     coll = ObstacleWithKinematicsCollection([Ts1, Ts2, Ts3])
    
#     fig2 = plt.figure(2)
#     ax = fig2.add_axes(111, projection='3d')
#     for t in np.linspace(0, 32, 32):
#         coll.plot3(t, ax=ax)
#     plt.show()

def test():
    enveloppe = ShipEnveloppe()
    enveloppe.rotate_and_translate(5., 10., 90.)
    print(enveloppe)
    enveloppe.reset()
    print(enveloppe)

if __name__ == "__main__":
    test()