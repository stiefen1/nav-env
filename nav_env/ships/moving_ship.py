"""
Goal: Create an intermediate class between MovingObstacle and ShipWithDynamicsBase | SailingShip that handles plot features and other stuff

"""
from nav_env.obstacles.obstacles import MovingObstacle
from nav_env.ships.states import States3
# from physics import ShipPhysics
from nav_env.ships.enveloppe import ShipEnveloppe
from nav_env.obstacles.obstacles import Obstacle
from typing import Callable
from random import randint
from nav_env.control.path import TimeStampedWaypoints as TSWPT
from seacharts import ENC
import matplotlib.colors as mat_colors
import matplotlib.pyplot as plt


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
                 mmsi:str=None,
                 **kwargs
                 ):
        
        if mmsi is None:
            # We do not use 0 and 9 to avoid bugs related to AIS database:
            # We use mmsi as a base number when showing multiple times the same ship. For every timestamp
            # we increase the base number, if we are close to 999999999 it can lead to a bug.
            # On the other hand, if the first digit is a '0', converting it into an int, and then again
            # into a string will lead to a mmsi with 8 digits.
            mmsi = ''.join(["{}".format(randint(1, 8)) for _ in range(0, 9)])
        assert len(mmsi) == 9, f"mmsi must be 9-digits number but has only {len(mmsi)} digits"
        self._mmsi = mmsi

        enveloppe = ShipEnveloppe(length=length, width=width, ratio=ratio, **kwargs)
        self._length = length
        self._width = width
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
        return super().plot(*args, ax=ax, params=params, c=c, **kwargs)

    def plot_traj_to_enc(self, enc:ENC, times:list, colormap:str='viridis', alpha=1., width:float=None, thickness:float=None, edge_style:str | tuple=None, marker_type:str=None) -> None:
        # Compute colors
        t_min = min(times)
        t_max = max(times)
        norm = mat_colors.Normalize(vmin=t_min, vmax=t_max)
        cmap = plt.get_cmap(colormap)

        t_prev = None
        wpt_prev = None
        for t in times:
            wpt = self.pose_fn(t).xy
            if t_prev is None:
                t_prev = t
                wpt_prev = wpt
                continue
            # Get corresponding color
            t_mean = (t+t_prev)/2
            rgb = cmap(norm(t_mean))[:3]
            color = mat_colors.to_hex((*rgb, alpha), keep_alpha=True)
            enc.display.draw_line([wpt_prev, wpt], color=color, width=width, thickness=thickness, edge_style=edge_style, marker_type=marker_type)
            t_prev = t
            wpt_prev = wpt
        return None

    def export_future_to_database(
        self,
        times:list,
        timestamps:list,
        path_to_database:str,
        colormap:str='viridis',
        alpha:float=1.,
        table:str='AisHistory',
        heading_in_seacharts_frame:bool=True,
        clear_table:bool=False,
        scale:float=1.,
        isolate_timestamps:bool=False
    ) -> None:
        return TSWPT.from_trajectory_fn(self.pose_fn, times).to_sql(path_to_database=path_to_database,
                                                                  mmsi=int(self.mmsi),
                                                                  timestamps=timestamps,
                                                                  colormap=colormap,
                                                                  alpha=alpha,
                                                                  table=table,
                                                                  heading_in_seacharts_frame=heading_in_seacharts_frame,
                                                                  clear_table=clear_table,
                                                                  length=self.length,
                                                                  width=self.width,
                                                                  scale=scale,
                                                                  isolate_timestamps=isolate_timestamps
                                                                  )

    @property
    def mmsi(self) -> str:
        return self._mmsi
    
    @property
    def width(self) -> float:
        return self._width
    
    @property
    def length(self) -> float:
        return self._length