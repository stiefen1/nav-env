"""
Goal: Create an intermediate class between MovingObstacle and ShipWithDynamicsBase | SailingShip that handles plot features and other stuff

"""
from nav_env.obstacles.obstacles import MovingObstacle
from nav_env.ships.states import States3
# from physics import ShipPhysics
from nav_env.ships.sailing_ship import ShipEnveloppe
from nav_env.obstacles.obstacles import Obstacle
from typing import Callable

class MovingShip(MovingObstacle):
    def __init__(self,
                 states:States3,
                 length:float=None,
                 width:float=None,
                 ratio:float=None,
                 pose_fn: Callable[[float], States3]=None,
                 name:str="MovingShip",
                 domain:Obstacle=None,
                 domain_margin_wrt_enveloppe:float=0.,
                 dt:float=None,
                 id:int=None,
                 **kwargs
                 ):
        
        enveloppe = ShipEnveloppe(length=length, width=width, ratio=ratio, **kwargs)
        super().__init__(
            pose_fn=pose_fn,
            initial_state=states,
            xy=enveloppe.get_xy_as_list(),
            dt=dt,
            domain=domain,
            domain_margin_wrt_enveloppe=domain_margin_wrt_enveloppe,
            name=name,
            id=id
            )
        
    def plot(self, *args, ax=None, params={'enveloppe':1}, c='r', **kwargs):
        super().plot(*args, ax=ax, params=params, c=c, **kwargs)