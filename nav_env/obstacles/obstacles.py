from shapely import Polygon, Point
from nav_env.geometry.wrapper import GeometryWrapper
from typing import Callable
import matplotlib.pyplot as plt
from shapely import affinity
from nav_env.ships.states import States3
from nav_env.control.states import DeltaStates
from copy import deepcopy

DEFAULT_INTEGRATION_STEP = 0.1
class Obstacle(GeometryWrapper):
    def __init__(self, xy: list=None, polygon: Polygon=None, geometry_type: type=Polygon, img:str=None, id:int=None):
        self._id = id
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
    
    def plot(self, *args, ax=None, c='black', **kwargs):
        """
        Plot the obstacle.
        """
        return super().plot(*args, ax=ax, c=c, **kwargs)
    
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
        super().__init__(polygon=affinity.translate(affinity.scale(Point([0, 0]).buffer(1), b, a), y, x), id=id)
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

class MovingObstacle(Obstacle):
    """
    Model an obstacle that changes over time.
    """
    def __init__(self, pose_fn:Callable[[float], States3]=None,
                 initial_state:States3=None,
                 t0:float=0.,
                 dt:float=None,
                 xy: list=None,
                 polygon: Polygon=None,
                 geometry_type: type=Polygon,
                 domain:Obstacle=None,
                 domain_margin_wrt_enveloppe:float=0.,
                 name:str="MovingObstacle",
                 id:int=None):
        """
        pose_fn: Callable that returns the pose of the obstacle at a given time as a tuple (x, y, angle).
        """
        super().__init__(xy=xy, polygon=polygon, geometry_type=geometry_type, id=id)

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
        
        # Domain of the obstacle
        if domain is None:
            initial_centroid = self.centroid
            self._domain:Obstacle = Obstacle(polygon=self._geometry).buffer(domain_margin_wrt_enveloppe, join_style='mitre')
            new_centroid = self._domain.centroid
            self._domain.translate_inplace(initial_centroid[0] - new_centroid[0], initial_centroid[1] - new_centroid[1])
        else:
            assert isinstance(domain, Obstacle), f"Expected Obstacle got {type(domain).__name__}"
            self._domain = domain

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

    def plot(self, *args, ax=None, domain:bool=False, **kwargs):
        """
        Plot the obstacle.
        """
        ax = super().plot(*args, ax=ax, **kwargs)
        if domain:
            return self._domain.plot(*args, ax=ax, linestyle='dashed', **kwargs)
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
    
    def enveloppe_fn_from_current_state(self, t:float) -> Obstacle:
        """
        Compute the enveloppe at time t from the current states.
        """
        states_at_t = self.pose_fn_from_current_state(t)
        return Obstacle(polygon=self._initial_geometry).rotate_and_translate(states_at_t.x, states_at_t.y, states_at_t.psi_deg)
    
    def domain_fn_from_current_state(self, t:float) -> Obstacle:
        """
        Compute the domain at time t from the current states.
        """
        states_at_t = self.pose_fn_from_current_state(t)
        return self._initial_domain.rotate_and_translate(states_at_t.x, states_at_t.y, states_at_t.psi_deg, origin=self._initial_domain.centroid)
    
    def pose_fn(self, t) -> States3:
        return self._pose_fn(t)
    
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
    
    coll = MovingObstacleCollection([o1, o2, o3, o4])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tmin, tmax = 0,  6
    
    for t in np.linspace(tmin, tmax, 50):
        ax = coll.plot3(t, ax=ax, c='black', alpha=0.5, domain=True)

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

if __name__ == "__main__":
    # test_basic_obstacle()
    show_time_varying_obstacle_as_3d()