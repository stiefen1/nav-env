"""
Ideal usage:
s1 = ShipObstacle(length=l, width=w, p0=p0, v0=v0)
s2 = ShipObstacle(domain=domain, p_t=lambda t: p(t))

"""
from nav_env.ships.states import States3, States2
from nav_env.obstacles.obstacles import Obstacle
from typing import Callable
from math import atan2, pi
from nav_env.ships.moving_ship import MovingShip


# TODO: Make possible to use SailingShip (Guided by a pose function) as own_ship in NavigationEnvironment

class SailingShip(MovingShip):
    """
    A target ship that moves according to either:
    - a given pose function p_t: t -> (x, y, heading)
    - a given initial position p0 and speed v0
    The flag make_heading_consistent allows to make the heading consistent with the trajectory, i.e. aligned with the velocity vector.
    """
    def __init__(self,
                 length: float=None,
                 width: float=None,
                 ratio: float=None,
                 pose_fn: Callable[[float], States3]=None,
                 initial_state: States2 | States3 | tuple=None,
                 domain:Obstacle=None,
                 domain_margin_wrt_enveloppe=0., 
                 dt:float=None,
                 id:int=None,
                 name:str="SailingShip",
                 **kwargs
                 ):
        
        """
        If issues with the pose_fn, try to define the function (using def, not lambda) outside of the test() function.
        """
            
        if initial_state is None:
            pass
        elif isinstance(initial_state, tuple) and len(initial_state) == 4:
            initial_state = States3(initial_state[0], initial_state[1], 0, initial_state[2], initial_state[3], 0)
        elif isinstance(initial_state, tuple) and len(initial_state) == 6:
            initial_state = States3(*initial_state)
        elif isinstance(initial_state, States2):
            initial_state = States3(*initial_state.xy, 0, *initial_state.xy_dot, 0)
        elif isinstance(initial_state, States3):
            pass
        else:
            raise ValueError(f"Expected States2 or States3 for initial_state, got {type(initial_state).__name__}")
        
        super().__init__(
            pose_fn=pose_fn,
            length=length,
            width=width,
            ratio=ratio,
            states=initial_state,
            domain=domain,
            id=id,
            dt=dt,
            domain_margin_wrt_enveloppe=domain_margin_wrt_enveloppe,
            name=name
        )

    # def plot(self, *args, ax=None, params:dict={'envelsoppe':1}, **kwargs):
    #     super().plot(*args, ax=ax, params=params, **kwargs)

    # def draw(self, *args, params:dict={'enveloppe':1}, **kwargs):
    #     super().draw(*args, params=params, **kwargs)

    def pose_fn(self, t:float) -> States3:
        """
        Override get_pose_at to make the heading consistent with the trajectory.
        """
        dt = 1e-2
        pose_at_t1 = self._pose_fn(t)
        pose_at_t2 = self._pose_fn(t+dt)
        dxdt = (pose_at_t2.x - pose_at_t1.x) / dt
        dydt = (pose_at_t2.y - pose_at_t1.y) / dt
        heading = atan2(dydt, dxdt) * 180 / pi - 90
        return States3(pose_at_t1.x, pose_at_t1.y, heading, pose_at_t1.x_dot, pose_at_t1.y_dot, pose_at_t1.psi_dot_deg)    

def test():
    import matplotlib.pyplot as plt
    from nav_env.obstacles.collection import MovingObstacleCollection
    from nav_env.obstacles.obstacles import Ellipse
    import numpy as np
    from math import cos, sin

    p = lambda t: States3(x=-10*cos(0.2*t), y=8*sin(0.2*t))
    Ts1 = SailingShip(pose_fn=p, domain=Ellipse(0., 0., 5., 10.))
    Ts2 = SailingShip(length=30, width=10, ratio=7/9, initial_state=States2(0, 0, 1, 1))
    Ts3 = SailingShip(width=8, ratio=3/7, pose_fn=lambda t: States3(t, -t, t*10))
    coll = MovingObstacleCollection([Ts1, Ts2, Ts3])
    
    fig2 = plt.figure(2)
    ax = fig2.add_axes(111, projection='3d')
    for t in np.linspace(0, 32, 32):
        coll.plot3(t, ax=ax, c='black', alpha=0.5, domain=True)
    plt.show()

if __name__ == "__main__":
    test()