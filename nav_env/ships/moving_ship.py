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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import cos, sin, pi
from nav_env.sensors.collection import SensorCollection



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
                 du:float=0.0,      # Uncertainty in surge speed
                 dpsi:float=0.0,    # Uncertainty in psi angle
                 sensors:SensorCollection|list=SensorCollection.empty(),
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
        self.du = du
        self.dpsi = dpsi
        self.sensors = sensors if isinstance(sensors, SensorCollection) else SensorCollection(sensors)
        self.sensors.system = self 
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
    
    def robust_polyhedron(self, t:float, *args, **kwargs) -> Obstacle:
        u = self.states.u
        x = self.states.x
        y = self.states.y
        psi_rad, psi_deg = self.states.psi_rad, self.states.psi_deg
        dpsi = self.dpsi
        du = self.du

        # Extract virtual ships used to build robust envelope
        ship1 = self._initial_domain.rotate_and_translate(-(u+du)*sin(psi_rad+dpsi/2)*t, (u+du)*cos(psi_rad+dpsi/2)*t, psi_deg+dpsi*180/pi/2).translate(x, y)
        ship2 = self._initial_domain.rotate_and_translate(-(u-du)*sin(psi_rad-dpsi)*t, (u-du)*cos(psi_rad-dpsi)*t, psi_deg-dpsi*180/pi).translate(x, y)
        ship3 = self._initial_domain.rotate_and_translate(-(u-du)*sin(psi_rad)*t, (u-du)*cos(psi_rad)*t, psi_deg).translate(x, y)
        ship4 = self._initial_domain.rotate_and_translate(-(u-du)*sin(psi_rad+dpsi)*t, (u-du)*cos(psi_rad+dpsi)*t, psi_deg+dpsi*180/pi).translate(x, y)
        ship5 = self._initial_domain.rotate_and_translate(-(u+du)*sin(psi_rad-dpsi)*t, (u+du)*cos(psi_rad-dpsi)*t, psi_deg-dpsi*180/pi).translate(x, y)
        ship6 = self._initial_domain.rotate_and_translate(-(u+du)*sin(psi_rad)*t, (u+du)*cos(psi_rad)*t, psi_deg).translate(x, y)
        ship7 = self._initial_domain.rotate_and_translate(-(u+du)*sin(psi_rad+dpsi)*t, (u+du)*cos(psi_rad+dpsi)*t, psi_deg+dpsi*180/pi).translate(x, y)
        ship8 = self._initial_domain.rotate_and_translate(-(u+du)*sin(psi_rad-dpsi/2)*t, (u+du)*cos(psi_rad-dpsi/2)*t, psi_deg-dpsi*180/pi/2).translate(x, y)
        
        # Extract vertices of robust envelope
        p0 = ship6.get_xy_as_list()[0]
        p1 = ship1.get_xy_as_list()[0]
        p2 = ship7.get_xy_as_list()[0]
        p3 = ship7.get_xy_as_list()[4]
        p4 = ship4.get_xy_as_list()[3]
        p50, p51 = ship3.get_xy_as_list()[2], ship3.get_xy_as_list()[3]
        p5 = ((p50[0]+p51[0])/2, (p50[1]+p51[1])/2)
        p6 = ship2.get_xy_as_list()[2]
        p7 = ship5.get_xy_as_list()[1]
        p8 = ship5.get_xy_as_list()[0]
        p9 = ship8.get_xy_as_list()[0]
        return Obstacle(xy=[p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])


    def plot3_polyhedron(self, t0, tf, *args, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        xy0 = self.robust_polyhedron(t0).xy
        xyf = self.robust_polyhedron(tf).xy
        z0 = [t0]*len(xy0[0])
        zf = [tf]*len(xyf[0])

        for i, (xyz0, xyzf) in enumerate(zip(zip(xy0[0], xy0[1], z0), zip(xyf[0], xyf[1], zf))):            
            if i==0:
                xyz0_prev = xyz0
                xyzf_prev = xyzf
                continue

            polygon = Poly3DCollection([[xyz0_prev, xyz0, xyzf, xyzf_prev, xyz0_prev]], *args, **kwargs)
            ax.add_collection(polygon)

            xyz0_prev = xyz0
            xyzf_prev = xyzf

        polygon_at_t0 = Poly3DCollection([list(zip(xy0[0], xy0[1], z0))], *args, **kwargs)
        polygon_at_tf = Poly3DCollection([list(zip(xyf[0], xyf[1], zf))], *args, **kwargs)

        ax.add_collection(polygon_at_t0)
        ax.add_collection(polygon_at_tf)

        return ax

    # def plot3_robust_polyhedron(self, t0:float, tf:float, *args, ax=None, **kwargs):
    #     """
    #     Plot the obstacle in 3D as a robust polyhedron
    #     """
    #     if ax is None:
    #         _, ax = plt.subplots(subplot_kw={'projection': '3d'})

    #     states_at_t0:States3 = self.pose_fn(t0)
    #     xy0 = Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_at_t0.x, states_at_t0.y, states_at_t0.psi_deg).xy
    #     z0 = [t0]*len(xy0[0])

    #     states_at_tf:States3 = self.pose_fn(tf)
    #     xyf = Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_at_tf.x, states_at_tf.y, states_at_tf.psi_deg).xy
    #     zf = [tf]*len(xy0[0])

    #     for i, (xyz0, xyzf) in enumerate(zip(zip(xy0[0], xy0[1], z0), zip(xyf[0], xyf[1], zf))):            
    #         if i==0:
    #             xyz0_prev = xyz0
    #             xyzf_prev = xyzf
    #             continue

    #         polygon = Poly3DCollection([[xyz0_prev, xyz0, xyzf, xyzf_prev, xyz0_prev]], *args, **kwargs)
    #         ax.add_collection(polygon)

    #         xyz0_prev = xyz0
    #         xyzf_prev = xyzf

    #     polygon_at_t0 = Poly3DCollection([list(zip(xy0[0], xy0[1], z0))], *args, **kwargs)
    #     polygon_at_tf = Poly3DCollection([list(zip(xyf[0], xyf[1], zf))], *args, **kwargs)

    #     ax.add_collection(polygon_at_t0)
    #     ax.add_collection(polygon_at_tf)


    #     return ax

    @property
    def mmsi(self) -> str:
        return self._mmsi
    
    @property
    def width(self) -> float:
        return self._width
    
    @property
    def length(self) -> float:
        return self._length
    
def test_3d_polyhedron() -> None:
    ship = MovingShip(States3(psi_deg=-30, x_dot=1, y_dot=1.5))
    ax = ship.plot3_polyhedron(0, 20, alpha=0.3)
    ax.set_xlim((-50, 50))
    ax.set_ylim((-50, 50))
    ax.set_zlim((0, 30))
    plt.show()

def test_robust_polyhedron() -> None:
    from math import cos, sin, pi
    
    u = 5
    psi = pi/4
    ship = MovingShip(States3(x=-20, y=-200, psi_deg=psi*180/pi, x_dot=-u*sin(psi), y_dot=u*cos(psi)))
    dt = 60
    # u = (ship.states.x_dot**2 + ship.states.y_dot**2)**0.5
    dpsi = 5*pi/180
    du = 0.2
    ship0 = ship._initial_domain.rotate(psi*180/pi).translate(ship.states.x, ship.states.y)
    ship1 = ship._initial_domain.rotate_and_translate(-(u+du)*sin(ship.states.psi_rad+dpsi/2)*dt, (u+du)*cos(ship.states.psi_rad+dpsi/2)*dt, ship.states.psi_deg+dpsi*180/pi/2).translate(ship.states.x, ship.states.y)
    ship2 = ship._initial_domain.rotate_and_translate(-(u-du)*sin(ship.states.psi_rad-dpsi)*dt, (u-du)*cos(ship.states.psi_rad-dpsi)*dt, ship.states.psi_deg-dpsi*180/pi).translate(ship.states.x, ship.states.y)
    ship3 = ship._initial_domain.rotate_and_translate(-(u-du)*sin(ship.states.psi_rad)*dt, (u-du)*cos(ship.states.psi_rad)*dt, ship.states.psi_deg).translate(ship.states.x, ship.states.y)
    ship4 = ship._initial_domain.rotate_and_translate(-(u-du)*sin(ship.states.psi_rad+dpsi)*dt, (u-du)*cos(ship.states.psi_rad+dpsi)*dt, ship.states.psi_deg+dpsi*180/pi).translate(ship.states.x, ship.states.y)
    ship5 = ship._initial_domain.rotate_and_translate(-(u+du)*sin(ship.states.psi_rad-dpsi)*dt, (u+du)*cos(ship.states.psi_rad-dpsi)*dt, ship.states.psi_deg-dpsi*180/pi).translate(ship.states.x, ship.states.y)
    ship6 = ship._initial_domain.rotate_and_translate(-(u+du)*sin(ship.states.psi_rad)*dt, (u+du)*cos(ship.states.psi_rad)*dt, ship.states.psi_deg).translate(ship.states.x, ship.states.y)
    ship7 = ship._initial_domain.rotate_and_translate(-(u+du)*sin(ship.states.psi_rad+dpsi)*dt, (u+du)*cos(ship.states.psi_rad+dpsi)*dt, ship.states.psi_deg+dpsi*180/pi).translate(ship.states.x, ship.states.y)
    ship8 = ship._initial_domain.rotate_and_translate(-(u+du)*sin(ship.states.psi_rad-dpsi/2)*dt, (u+du)*cos(ship.states.psi_rad-dpsi/2)*dt, ship.states.psi_deg-dpsi*180/pi/2).translate(ship.states.x, ship.states.y)
    
    ax = ship0.plot()
    ship2.plot(ax=ax)
    ship3.plot(ax=ax)
    ship4.plot(ax=ax)
    ship5.plot(ax=ax)
    ship6.plot(ax=ax)
    ship7.plot(ax=ax)

    p0 = ship6.get_xy_as_list()[0]
    p1 = ship1.get_xy_as_list()[0]
    p2 = ship7.get_xy_as_list()[0]
    p3 = ship7.get_xy_as_list()[4]
    p4 = ship4.get_xy_as_list()[3]
    p50, p51 = ship3.get_xy_as_list()[2], ship3.get_xy_as_list()[3]
    p5 = ((p50[0]+p51[0])/2, (p50[1]+p51[1])/2)
    p6 = ship2.get_xy_as_list()[2]
    p7 = ship5.get_xy_as_list()[1]
    p8 = ship5.get_xy_as_list()[0]
    p9 = ship8.get_xy_as_list()[0]
    obs = Obstacle(xy=[p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])
    obs.fill(alpha=0.3, c='red', ax=ax)
    

    # p2 = ship
    ax.scatter(*p0, c='red')
    ax.scatter(*p1, c='red')
    ax.scatter(*p2, c='red')
    ax.scatter(*p3, c='red')
    ax.scatter(*p4, c='red')
    ax.scatter(*p5, c='red')
    ax.scatter(*p6, c='red')
    ax.scatter(*p7, c='red')
    ax.scatter(*p8, c='red')
    ax.scatter(*p9, c='red')

    ax.set_xlim((-300, 5))
    ax.set_ylim((-5, 300))
    ax.set_aspect('equal')
    plt.show()

def test_robust_polyhedron_from_self() -> None:
    
    u = 10
    psi = pi/4
    # ship = MovingShip(States3(x=-20, y=20, psi_deg=psi*180/pi, x_dot=-u*sin(psi), y_dot=u*cos(psi)), dpsi=5*pi/180, du=0.2)
    ship = MovingShip(States3(x=-20, y=20, psi_deg=psi*180/pi, u=u), dpsi=5*pi/180, du=0.2)

    dt = 20

    obs10 = ship.robust_polyhedron(10)
    obs30 = ship.robust_polyhedron(30)
    ax = ship.plot()
    obs10.fill(alpha=0.3, c='red', ax=ax)
    obs30.fill(alpha=0.3, c='red', ax=ax)

    v0x, v0y = zip(*obs10.get_xy_as_list())
    v10x, v10y = zip(*obs30.get_xy_as_list())

    print(v0x, v10x)

    

    ax.set_xlim((-300, 5))
    ax.set_ylim((-5, 300))
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    # test_3d_polyhedron()
    # test_robust_polyhedron()
    test_robust_polyhedron_from_self()