from shapely import Polygon, Point
from nav_env.geometry.wrapper import GeometryWrapper
from typing import Callable
import matplotlib.pyplot as plt
from shapely import affinity
from nav_env.ships.states import States3
from nav_env.control.states import DeltaStates
from copy import deepcopy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from math import pi

DEFAULT_INTEGRATION_STEP = 0.1
class Obstacle(GeometryWrapper):
    def __init__(self, xy: list=None, polygon: Polygon=None, geometry_type: type=Polygon, img:str=None, id:int=None, depth:float=None):
        self._id = id
        self._depth = depth
        super().__init__(xy=xy, polygon=polygon, geometry_type=geometry_type)

    def distance_to_obstacle(self, x:float, y:float) -> float:
        """
        Get the distance to the obstacle at a given position.
        """
        assert isinstance(self._geometry, Polygon), f"Obstacle must be a polygon not a {type(self._geometry)}"
        p = Point([x, y])
        if p.within(self._geometry):
            return -p.distance(self._geometry.exterior)
        return p.distance(self._geometry.exterior)
    
    def __repr__(self):
        return f"Obstacle({self.centroid[0]:.2f}, {self.centroid[1]:.2f})"
    
    def plot(self, *args, ax=None, c='black', offset:tuple=None, **kwargs):
        """
        Plot the obstacle.
        """
        return super().plot(*args, ax=ax, c=c, offset=offset, **kwargs)
    
    def fill(self, *args, ax=None, c='black', **kwargs):
        """
        Fill the obstacle.
        """
        return super().fill(*args, ax=ax, c=c, **kwargs)
    
    def plot3(self, z:float, *args, ax=None, **kwargs):
        """
        Plot the obstacle in 3D.
        """
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        z = [z]*len(self.xy[0])
        ax.plot(*self.xy, z, *args, **kwargs)
        return ax
    
    def project_on_spatio_temporal_plane(self, spatio_temporal_plane) -> "Obstacle":
        obs = deepcopy(self)
        if type(obs) is Obstacle:
            # Convert obstacle into MovingObstacle with zero speed
            obs = MovingObstacle(pose_fn=lambda t: States3(), xy=obs.get_xy_as_list(), id=obs.id)
        intersection = spatio_temporal_plane.project_on_euclidean(obs)
        return Obstacle(intersection)
    
    @property
    def id(self) -> int:
        return self._id
    
    @property
    def depth(self) -> float:
        return self._depth
    
class Ellipse(Obstacle):
    def __init__(self,
                x:float,
                y:float,
                a:float,
                b:float,
                # da:float=0,
                # db:float=0,
                id:int=None
                ):
        super().__init__(polygon=affinity.translate(affinity.scale(Point([0, 0]).buffer(1), b, a), x, y), id=id)
        self._a = a
        self._b = b
        self._da = x
        self._db = y

    @property
    def a(self):
        return self._a
    
    @property
    def b(self):
        return self._b
    
    @property
    def center(self):
        return self.centroid
    
    # @a.setter
    # def a(self, value):
    #     self._a = value
    #     self._geometry = affinity.scale(Point(self.center).buffer(1), self._a, self._b)
    
    # @b.setter
    # def b(self, value):
    #     self._b = value
    #     self._geometry = affinity.scale(Point(self.center).buffer(1), self._a, self._b)
    
    @center.setter
    def center(self, value:tuple):
        self.centroid = value

    @property
    def da(self) -> float:
        return self._da
    
    @property
    def db(self) -> float:
        return self._db
    
    def __repr__(self):
        return f"Ellipse({self.a:.2f}, {self.b:.2f} at {self.center[0]:.2f}, {self.center[1]:.2f})"
    
class Circle(Ellipse):
    def __init__(self, x, y, radius, id:int=None):
        super().__init__(x, y, radius, radius, id=id)
        self._radius = radius

    def __repr__(self):
        return f"Circle(at {self.center[0]:.2f}, {self.center[1]:.2f} with radius {self.radius:.2f})"

    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        self._radius = value
        self._geometry = affinity.scale(Point(self.center).buffer(1), value, value)

class Rectangle(Obstacle):
    def __init__(self, x, y, height, width, id:int=None): # 3 -5 2 4
        self._center = (x, y)
        self._dim = (width, height)
        super().__init__(xy=self.get_envelope_coordinates())
        
    def get_envelope_coordinates(self) -> list[tuple]:
        # Compute vertices coordinates
        v1 = (self._center[0] + self._dim[0]/2 , self._center[1] + self._dim[1] / 2)
        v2 = (self._center[0] + self._dim[0]/2 , self._center[1] - self._dim[1] / 2)
        v3 = (self._center[0] - self._dim[0]/2 , self._center[1] - self._dim[1] / 2)
        v4 = (self._center[0] - self._dim[0]/2 , self._center[1] + self._dim[1] / 2)
        return [v1, v2, v3, v4]

class MovingObstacle(Obstacle):
    """
    Model an obstacle that changes over time.
    """
    def __init__(self,
                 pose_fn:Callable[[float], States3]=None,
                 initial_state:States3=None,
                 t0:float=0.,
                 dt:float=None,
                 xy: list=None,
                 polygon: Polygon=None,
                 geometry_type: type=Polygon,
                 domain:Obstacle=None,
                 domain_margin_wrt_enveloppe:float=0.,
                 name:str="MovingObstacle",
                 id:int=None,
                 start_at:float=0.0
                 ):
        """
        pose_fn: Callable that returns the pose of the obstacle at a given time as a tuple (x, y, angle).
        """
        super().__init__(xy=xy, polygon=polygon, geometry_type=geometry_type, id=id)
        self.start_at = start_at
        # If a pose_fn is not provided, we use the initial states to compute the pose as x(t) = x0 + v * t
        if pose_fn is not None:
            self._pose_fn = pose_fn
            self._states = self.pose_fn(t0)
            self._initial_states = self.pose_fn(t0)
        else:
            self._initial_states = initial_state
            self._pose_fn = self.pose_fn_from_initial_state
            self._states = self._initial_states

        # At this point, _pose_fn, _states and _initial_states are set
 
        assert isinstance(self.pose_fn(0), States3), f"The pose function must return a States3 object"

        # Initial states
        self._t0 = t0
        self._t = t0
        self._dt = dt or DEFAULT_INTEGRATION_STEP
        self._name = name
        self._logs = {"times": np.zeros((0, 1)), "states": np.zeros((0, 6))}
        
        # Domain of the obstacle
        if domain is None:
            self.initial_centroid = self.centroid
            self._domain:Obstacle = Obstacle(polygon=self._geometry).buffer(domain_margin_wrt_enveloppe, join_style='mitre')
            new_centroid = self._domain.centroid
            self._domain.translate_inplace(self.initial_centroid[0] - new_centroid[0], self.initial_centroid[1] - new_centroid[1])
        else:
            assert isinstance(domain, Obstacle), f"Expected Obstacle got {type(domain).__name__}"
            self._domain = domain
            self.initial_centroid = self._domain.centroid

        # Initial geometry, could be avoided but it makes the things simpler
        self._initial_geometry = self._geometry
        self._initial_domain = deepcopy(self._domain)

        # Since we use the step() method to update the obstacle, we need to set the initial orientation of the obstacle
        prev_center = self.centroid
        self.rotate_and_translate_inplace(self._initial_states.x, self._initial_states.y, self._initial_states.psi_deg) # Change geometry (enveloppe)
        self._domain.rotate_and_translate_inplace(self._initial_states.x, self._initial_states.y, self._initial_states.psi_deg, origin=prev_center) # Change geometry (enveloppe)

    def step(self, *args, **kwargs) -> None:
        """
        Step the obstacle.
        """
        self._t += self._dt
        prev_state = self._states
        self._states = self.pose_fn(self._t)
        ds:DeltaStates = self._states - prev_state
        prev_center = self.centroid
        self.rotate_and_translate_inplace(ds.x, ds.y, ds.psi_deg) # Change geometry (enveloppe)
        self._domain.rotate_and_translate_inplace(ds.x, ds.y, ds.psi_deg, origin=prev_center) # Change geometry (enveloppe)
        self.save()

    def save(self) -> None:
        self.save_time()
        self.save_state()

    def save_time(self) -> None:
        self._logs["times"] = np.append(self._logs["times"], np.array(self._t).reshape(1, 1), axis=0)

    def save_state(self) -> None:
        self._logs["states"] = np.append(self._logs["states"], np.array([*self._states.pose, *self._states.uvr]).reshape(1, 6), axis=0)
    
    def get_envelope_from_logs_at_idx(self, idx:int) -> Obstacle:
        initial_centroid = self.initial_centroid
        dx = (
            self._logs["states"][idx, 0],
            self._logs["states"][idx, 1],
            180*(self._logs["states"][idx, 2])/pi
        )

        new_envelope = Obstacle(polygon=self._initial_geometry).rotate_and_translate(*dx, origin=initial_centroid)
        return new_envelope

    def get_domain_from_logs_at_idx(self, idx:int) -> Obstacle:
        initial_centroid = self._initial_domain.centroid
        dx = (
            self._logs["states"][idx, 0],
            self._logs["states"][idx, 1],
            180*(self._logs["states"][idx, 2])/pi
        )

        new_domain = self._initial_domain.rotate_and_translate(*dx, origin=initial_centroid)
        return new_domain

    # def get_geometry_from_logs_at_t(self, idx:int) -> Obstacle:
    #     initial_centroid = self._initial_geometry.centroid
    #     dx = (
    #         self._logs["states"][idx, 0] - self._initial_states.x,
    #         self._logs["states"][idx, 1] - self._initial_states.y,
    #         self._logs["states"][idx, 2] - self._initial_states.psi_rad
    #     )
    #     # self.rotate_and_translate_inplace(dx[0], dx[1], dx[2], use_radians=True)
    #     new_geometry = self._initial_domain.rotate_and_translate(dx[0], dx[1], dx[2], origin=initial_centroid, use_radians=True)
    #     return Obstacle(new_domain.xy)

    def reset(self) -> None:
        """
        Reset the obstacle.
        """
        self._t = self._t0
        self._states = deepcopy(self._initial_states)
        self._geometry = self._initial_geometry
        self._domain = deepcopy(self._initial_domain)
        prev_center = self.centroid
        self.rotate_and_translate_inplace(self._initial_states.x, self._initial_states.y, self._initial_states.psi_deg) # Change geometry (enveloppe)
        self._domain.rotate_and_translate_inplace(self._initial_states.x, self._initial_states.y, self._initial_states.psi_deg, origin=prev_center) # Change geometry (enveloppe)

    def plot(self, *args, ax=None, params:dict={'enveloppe':1}, domain:bool=False, **kwargs):
        """
        Plot the obstacle.
        """
        if ax is None:
            _, ax = plt.subplots()

        keys=params.keys()
        if 'enveloppe' in keys:
            ax = super().plot(*args, ax=ax, **kwargs)
        if 'domain' in keys:
            ax = self._domain.plot(*args, ax=ax, linestyle='dashed', **kwargs)
        if 'name' in keys:
            ax.text(*self._states.xy, self._name, fontsize=8, c='black')
        if 'ghost' in keys:
            """
            plot the ghost ship enveloppe at different times, assuming speed is constant
            """
            times = params['ghost']
            if isinstance(times, int):
                times = [times]
            for t in times:
                self.enveloppe_fn_from_current_state(t).plot(ax=ax, c='r', alpha=0.3)
        return ax

    def plot3(self, t:float, *args, ax=None, domain:bool=False, **kwargs):
        """
        Plot the obstacle in 3D.
        """
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        states_at_t:States3 = self.pose_fn(t)
        xy = Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_at_t.x, states_at_t.y, states_at_t.psi_deg).xy
        z = [t]*len(xy[0])
        ax.plot(*xy, z, *args, **kwargs)

        if domain:
            xy = self._initial_domain.rotate_and_translate(states_at_t.x, states_at_t.y, states_at_t.psi_deg, origin=self._initial_domain.centroid).xy
            z = [t]*len(xy[0])
            ax.plot(*xy, z, *args, linestyle='dashed', **kwargs)

        return ax
    
    # def plot3_polyhedron_with_uncertainties(self, t:float, interval:dict, *args, ax=None, **kwargs):
    #     """
    #     Plot the obstacle in 3D while interval of confidence for both speed intensity and direction (in degrees)
    #     """
    #     if ax is None:
    #         _, ax = plt.subplots(subplot_kw={'projection': '3d'})

    #     # Get uncertainties as an interval of confidence, i.e. we assume v \in [v_min, v_max] and alpha \in [alpha_min, alpha_max]   
    #     dv, dalpha  = 0., 0.
    #     if 'speed' in interval.keys():
    #         dv = interval['speed']
    #     if 'direction' in interval.keys():
    #         dalpha = 3.14159/180 * interval['direction'] # alpha in radians
    #     alpha0 = self._initial_states.psi_rad
    #     R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    #     p_of_t = np.array([*self._pose_fn(t).xy]).T
    #     p0 = np.array([*self._pose_fn(self._t0).xy]).T
    #     n = np.array([-np.sin(alpha0), np.cos(alpha0)]).T
    #     xy_0 = self.xy
    #     z_0 = [self._t0]*len(xy_0[0])
        
    #     boundaries = [(-dv, -dalpha), (-dv, dalpha), (dv, dalpha), (dv, -dalpha)]
    #     for dv_i, dalpha_i in boundaries:
    #         p_i = R(dalpha_i) @ (p_of_t - p0 + n * dv_i * t) + p0
    #         states_i = States3(p_i[0], p_i[1], (alpha0+dalpha_i)*180/np.pi)
    #         xy_i = Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_i.x, states_i.y, states_i.psi_deg).xy
    #         z_i = [t]*len(xy_i[0])
        
    #         for j, (xyz_0j, xyz_ij) in enumerate(zip(zip(xy_0[0], xy_0[1], z_0), zip(xy_i[0], xy_i[1], z_i))):
    #             if j==0:
    #                 xyz0_prev = xyz_0j
    #                 xyzf_prev = xyz_ij
    #                 continue

    #             polygon = Poly3DCollection([[xyz0_prev, xyz_0j, xyz_ij, xyzf_prev, xyz0_prev]], *args, **kwargs)
    #             ax.add_collection(polygon)

    #             xyz0_prev = xyz_0j
    #             xyzf_prev = xyz_ij

    #         polygon_i = Poly3DCollection([list(zip(xy_i[0], xy_i[1], z_i))], *args, **kwargs)
    #         ax.add_collection(polygon_i)

    #     polygon_at_t0 = Poly3DCollection([list(zip(xy_0[0], xy_0[1], z_0))], *args, **kwargs)
    #     ax.add_collection(polygon_at_t0)
    #     return ax

    
    def plot3_polyhedron(self, t0:float, tf:float, *args, ax=None, **kwargs):
        """
        Plot the obstacle in 3D as a polyhedron
        """
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        states_at_t0:States3 = self.pose_fn(t0)
        xy0 = Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_at_t0.x, states_at_t0.y, states_at_t0.psi_deg).xy
        z0 = [t0]*len(xy0[0])

        states_at_tf:States3 = self.pose_fn(tf)
        xyf = Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_at_tf.x, states_at_tf.y, states_at_tf.psi_deg).xy
        zf = [tf]*len(xy0[0])

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

            
            

        # test = list(zip(, , ))
        # print(len(test), len(test[0]))
        # polygon = Poly3DCollection([test])
        # ax.add_collection(polygon)

        # ax.scatter(xy0[0] + xyf[0], xy0[1] + xyf[1], z0+zf)
        # ax.scatter(xyf[0], xyf[1], zf)

        return ax


    
    def quiver_speed(self, *args, ax=None, **kwargs):
        """
        Plot the speed of the obstacle.
        """
        if ax is None:
            _, ax = plt.subplots()
        states:States3 = self.pose_fn(1)
        states.plot(*args, ax=ax, **kwargs)
        return ax

    def __call__(self, t:float=None, domain:bool=False) -> Obstacle:
        if t is None:
            t = self._t
        states_at_t:States3 = self.pose_fn(t)

        if domain:
            return self._initial_domain.rotate_and_translate(states_at_t.x, states_at_t.y, states_at_t.psi_deg, origin=self._initial_domain.centroid)
        return Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_at_t.x, states_at_t.y, states_at_t.psi_deg)

    def __repr__(self):
        return f"MovingObstacle({self.centroid[0]:.2f}, {self.centroid[1]:.2f})"
    
    def pose_fn_from_initial_state(self, t) -> States3:
        initial_state = self._initial_states
        return States3(initial_state.x + initial_state.x_dot * t,
                       initial_state.y + initial_state.y_dot * t,
                       initial_state.psi_deg + initial_state.psi_dot_deg * t,
                       initial_state.x_dot,
                       initial_state.y_dot,
                       initial_state.psi_dot_deg)
    
    def pose_fn_from_current_state(self, t:float) -> States3:
        """
        Compute pose x_t at time t from the current states, assuming x(t) = x_t + v_t * t. Could be used to compute metrics such as DCPA, TCPA, etc.
        """
        current_state = self._states
        return States3(current_state.x + current_state.x_dot * t,
                          current_state.y + current_state.y_dot * t,
                          current_state.psi_deg + current_state.psi_dot_deg * t,
                          current_state.x_dot,
                          current_state.y_dot,
                          current_state.psi_dot_deg)
    
    def linear_prediction_from_current_state(self, t:float) -> States3:
        current_state = self._states
        return States3(current_state.x + current_state.x_dot * t,
                          current_state.y + current_state.y_dot * t,
                          current_state.psi_deg,
                          current_state.x_dot,
                          current_state.y_dot,
                          0.0)
    
    def linear_prediction_from_current_state_given_u_and_psi(self, t:float, u_d:float, psi_d:float) -> States3:
        current_state = self._states
        x_dot, y_dot = self.from_ship_to_world_frame(States3(x_dot=u_d), psi_d).xy_dot
        return States3(current_state.x + x_dot * t,
                          current_state.y + y_dot * t,
                          psi_d,
                          x_dot,
                          y_dot,
                          0)
    
    def pose_fn_from_current_state_given_u_and_psi(self, t:float, u_d:float, psi_d:float):
        """
        Primarily to be used with SB-MPC to generate scenarios based on linear approximation of own ship
        given two parameters; formward speed and heading
        """
        
        current_state = self._states
        ### We have to make x_dot, y_dot consistent with u_d and psi_d, assuming v_d = 0
        # Convert u_d, v_d into world frame
        x_dot, y_dot = self.from_ship_to_world_frame(States3(x_dot=u_d), psi_d).xy_dot
        return States3(current_state.x + x_dot * t,
                          current_state.y + y_dot * t,
                          psi_d,
                          x_dot,
                          y_dot,
                          0)

    def from_world_to_ship_frame(self, states_in_world_frame:States3) -> States3:
        xypsi_dot = states_in_world_frame.to_numpy()[3:6]
        uvr = self.get_rotation_matrix(states_in_world_frame.psi_rad) @ xypsi_dot.T
        states_in_ship_frame = States3(*states_in_world_frame.pose_deg)
        states_in_ship_frame.x_dot = uvr[0]
        states_in_ship_frame.y_dot = uvr[1]
        states_in_ship_frame.psi_dot_deg = uvr[2]
        return states_in_ship_frame

    def from_ship_to_world_frame(self, states_in_ship_frame:States3, psi_rad:float) -> States3:
        uvr = states_in_ship_frame.to_numpy()[3:6]
        xypsi_dot = self.get_rotation_matrix(psi_rad).T @ uvr.T
        states_in_world_frame = States3(*states_in_ship_frame.pose_deg)
        states_in_world_frame.x_dot = xypsi_dot[0]
        states_in_world_frame.y_dot = xypsi_dot[1]
        states_in_world_frame.psi_dot_deg = xypsi_dot[2]
        return states_in_world_frame

    def get_rotation_matrix(self, psi_rad) -> np.ndarray:
        return np.array([
            [-np.sin(psi_rad), np.cos(psi_rad), 0.],
            [np.cos(psi_rad), np.sin(psi_rad), 0.],
            [0., 0., -1.]
        ])
    
    # @property
    # def uvr(self) -> tuple:
    #     xypsi_dot = self.states.to_numpy()[3:6]
    #     return tuple((self.get_rotation_matrix(self.states.psi_rad) @ xypsi_dot.T).tolist())

    def enveloppe_fn_from_current_state(self, t:float) -> Obstacle:
        """
        Compute the enveloppe at time t from the current states.
        """
        states_at_t = self.pose_fn_from_current_state(t)
        return Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_at_t.x, states_at_t.y, states_at_t.psi_deg)
    
    def enveloppe_fn_from_current_state_given_u_and_psi(self, t:float, u_d:float, psi_d:float) -> Obstacle:
        """
        Compute the enveloppe at time t from the current states.
        """
        states_at_t = self.pose_fn_from_current_state(t)
        return Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_at_t.x, states_at_t.y, states_at_t.psi_deg)
    
    def enveloppe_fn_from_linear_prediction(self, t:float) -> Obstacle:
        states_at_t = self.linear_prediction_from_current_state(t)
        return Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_at_t.x, states_at_t.y, states_at_t.psi_deg)
    
    def enveloppe_fn_from_linear_prediction_given_u_and_psi(self, t:float, u_d:float, psi_d:float) -> Obstacle:
        states_at_t = self.linear_prediction_from_current_state_given_u_and_psi(t, u_d, psi_d)
        return Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_at_t.x, states_at_t.y, states_at_t.psi_deg)
    

    def domain_fn_from_current_state(self, t:float) -> Obstacle:
        """
        Compute the domain at time t from the current states.
        """
        states_at_t = self.pose_fn_from_current_state(t)
        return self._initial_domain.rotate_and_translate(states_at_t.x, states_at_t.y, states_at_t.psi_deg, origin=self._initial_domain.centroid)
    
    def pose_fn(self, t) -> States3:
        if t >= self.start_at:
            return self._pose_fn(t-self.start_at)
        else:
            return self._pose_fn(0)
    
    @property
    def dt(self) -> float:
        return self._dt
    
    @dt.setter
    def dt(self, value:float) -> None:
        self._dt = value

    @property
    def domain(self) -> Obstacle:
        return self._domain

    @property
    def states(self) -> States3:
        return self._states
    
    @states.setter
    def states(self, value:States3) -> None:
        self._states = value
    
    @property
    def states_in_ship_frame(self) -> States3:
        return self.from_world_to_ship_frame(self._states)
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, value:str) -> None:
        self._name = value
    

def test_basic_obstacle():
    from matplotlib import pyplot as plt
    import time
    c = Circle(0, 0, 1)
    o = Obstacle(xy=[(0, 0), (2, 0), (2, 2), (0, 2)]).translate(0, -0.5).rotate(45)

    c.center = (0, -1)
    c.radius = 2
    ax = o.plot()
    c.plot(ax=ax)
    c.difference(o).plot()
    c.convex_hull_of_union(o).plot()
    plt.show()


def show_time_varying_obstacle_as_3d():
    from matplotlib import pyplot as plt
    import numpy as np
    from nav_env.obstacles.collection import MovingObstacleCollection, ObstacleCollection
    from nav_env.obstacles.obstacles import Obstacle
    from scipy.spatial.transform import Rotation as R

    o1 = MovingObstacle(initial_state=States3(0., 0., 45, -1., 2., 0.), xy=[(0, 0), (2, 0), (2, 2), (0, 2)])
    o2 = MovingObstacle(initial_state=States3(10., 5., 90., -2., -1., 0), xy=[(0, 0), (2, 0), (3, 1), (2, 2), (0, 2)])
    o3 = MovingObstacle(initial_state=States3(0., 5., 0., -1.2, -0.5, 0.), xy=[(0, 0), (2, 0), (3, 1), (2, 2), (0, 2)])
    o4 = MovingObstacle(initial_state=States3(-3., -5., 0., 2., 0.2, 0.), xy=[(0, 0), (2, 0), (3, 1), (2, 2), (0, 2)])
    
    # coll = MovingObstacleCollection([o1, o2, o3, o4])
    coll = MovingObstacleCollection([o1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tmin, tmax = 0,  6
    
    ax = coll.plot3_polyhedra_with_uncertainties(50, intervals={'speed': 1., 'direction': 5}, ax=ax)
    o1.plot3_polyhedron(0, 50, ax=ax)
    for t in np.linspace(tmin, tmax, 50):
        # ax = coll.plot3_polyhedra_with_uncertainties(t, intervals={'speed': 2., 'direction': 50}, ax=ax)
        pass
        # ax = o1.plot3_polyhedron()
        # ax = coll.plot3(t, ax=ax, c='black', alpha=0.5, domain=True)


    # Build plane
    x0, y0, t0 = -10, -10, 0
    xf, yf, tf = 10, 10, tmax
    b = ((x0 + y0) * tf - t0 * (xf + yf)) / (t0 - tf)
    a = (xf + yf + b) / tf

    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = (X + Y + b) / a
    Z[Z < 0] = np.nan

    # Calculate the angle of the plane and deduce the rotation matrix
    theta = np.arctan((tmax-t0)/(np.linalg.norm([xf-x0, yf-y0])))
    axis = np.array([1/np.sqrt(2), -1/np.sqrt(2), 0])
    r = R.from_rotvec(-theta * axis / np.linalg.norm(axis))

    # Intersection with obstacle 1:
    # print(o1(tmin).get_xy_as_list(), o1(tmax).xy[0])
    coll_2d = ObstacleCollection([])
    for obs in coll:
        obs_2d = np.empty(shape=(0, 2))
        for i, (p0, pf) in enumerate(zip(obs(tmin).buffer(0, join_style='mitre').get_xy_as_list(), obs(tmax).buffer(0, join_style='mitre').get_xy_as_list())):
            p0, pf = np.array(p0)[:, np.newaxis], np.array(pf)[:, np.newaxis]
            t = (np.ones((1, 2)) @ p0 + b) / (a - np.ones((1, 2)) @ ((pf-p0)/(tmax-tmin)))
            p_star = p0 + (pf - p0) * t / (tmax - tmin)
            ax.plot(p_star[0], p_star[1], t, 'ro')
            new_point_3d = np.array([p_star[0], p_star[1], t[0]]).T
            new_point_3d_proj = r.apply(new_point_3d-np.array([[x0], [y0], [t0]]).T) + np.array([[x0], [y0], [t0]]).T
            obs_2d = np.append(obs_2d, new_point_3d_proj[:, 0:2], axis=0)

        coll_2d.append(Obstacle(xy=obs_2d))

    # Bring back to 2D
    # obs_2d = r.apply(obs_3d-np.array([[x0], [y0], [t0]]).T) + np.array([[x0], [y0], [t0]]).T
    # print(obs_3d, obs_2d)

    # Plot the surface.
    ax.plot_surface(X, Y, Z, alpha=0.3,
                        linewidth=0, antialiased=False)
    
    
    # ax.plot(*obs_2d.T, 'ro')
    coll_2d.plot(ax=ax)
    ax.scatter([x0, xf], [y0, yf], c='g')
    
    
    
    ax.set_zlim(0, tmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time')
    plt.show()

def test_union_of_obstacles() -> None:
    from nav_env.obstacles.collection import ObstacleCollection
    import matplotlib.pyplot as plt
    o1 = Obstacle(xy=[(0, 0), (2, 0), (2, 2), (0, 2)])
    o2 = Obstacle(xy=[(0, 0), (2, 0), (3, 1), (2, 2), (0, 2)]).translate(1, -1)
    coll = ObstacleCollection([o1, o2])
    coll.group_intersecting_obstacles()
    coll.plot()
    plt.show()

    o12 = o1.union(o2)
    o12.plot()
    plt.show()

def test_rectangle() -> None:
    r = Rectangle(3, -5, 2, 4)
    r.plot()
    plt.show()

if __name__ == "__main__":
    # test_basic_obstacle()
    # show_time_varying_obstacle_as_3d()
    # test_union_of_obstacles()
    test_rectangle()