from shapely import Polygon, Point
from nav_env.geometry.wrapper import GeometryWrapper
from typing import Callable
import matplotlib.pyplot as plt
from shapely import affinity
from nav_env.ships.states import States3

class Obstacle(GeometryWrapper):
    def __init__(self, xy: list=None, polygon: Polygon=None, geometry_type: type=Polygon, img:str=None):
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
    
    def plot(self, ax=None, c='b', alpha=1, **kwargs):
        """
        Plot the obstacle.
        """
        return super().plot(ax=ax, c=c, alpha=alpha, **kwargs)

class Circle(Obstacle):
    def __init__(self, x, y, radius):
        super().__init__(polygon=Point([x, y]).buffer(radius))
        self._radius = radius

    @property
    def radius(self):
        return self._radius
    
    @property
    def center(self):
        return self.centroid
    
    @radius.setter
    def radius(self, value):
        self._radius = value
        self._geometry = Point(self.center).buffer(value)
    
    @center.setter
    def center(self, value:tuple):
        self.centroid = value
    
    def __repr__(self):
        return f"Circle({self.radius:.2f} at {self.center[0]:.2f}, {self.center[1]:.2f})"
    
class Ellipse(Obstacle):
    def __init__(self, x, y, a, b):
        super().__init__(polygon=affinity.scale(Point([x, y]).buffer(1), a, b))
        self._a = a
        self._b = b

    @property
    def a(self):
        return self._a
    
    @property
    def b(self):
        return self._b
    
    @property
    def center(self):
        return self.centroid
    
    @a.setter
    def a(self, value):
        self._a = value
        self._geometry = affinity.scale(Point(self.center).buffer(1), self._a, self._b)
    
    @b.setter
    def b(self, value):
        self._b = value
        self._geometry = affinity.scale(Point(self.center).buffer(1), self._a, self._b)
    
    @center.setter
    def center(self, value:tuple):
        self.centroid = value
    
    def __repr__(self):
        return f"Ellipse({self.a:.2f}, {self.b:.2f} at {self.center[0]:.2f}, {self.center[1]:.2f})"

class ObstacleWithKinematics(Obstacle):
    """
    Model an obstacle that changes over time.
    """
    def __init__(self, pose_fn:Callable=None, initial_state:States3=None, xy: list=None, polygon: Polygon=None, geometry_type: type=Polygon):
        """
        pose_fn: Callable that returns the pose of the obstacle at a given time as a tuple (x, y, angle).
        """
        super().__init__(xy=xy, polygon=polygon, geometry_type=geometry_type)
        if pose_fn is not None:
            self._pose_fn = pose_fn
        else:
            self._initial_state = initial_state
            self._pose_fn = self.get_pose_at

    def plot3(self, t:float, ax=None, c='b', alpha=0.3, **kwargs):
        """
        Plot the obstacle in 3D.
        """
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        x, y, angle = self._pose_fn(t)
        xy = self.rotate_and_translate(x, y, angle).xy
        z = [t]*len(xy[0])
        ax.plot(*xy, z, c=c, alpha=alpha, **kwargs)
        return ax
    
    def quiver_speed(self, ax=None, c='b', **kwargs):
        """
        Plot the speed of the obstacle.
        """
        if ax is None:
            _, ax = plt.subplots()
        xdot, ydot, _ = self._pose_fn(1)
        x, y = self.centroid
        ax.quiver(x, y, xdot, ydot, color=c, **kwargs)
        return ax

    def __call__(self, t:float) -> Obstacle:
        return Obstacle(polygon=self._geometry).rotate_and_translate(*self._pose_fn(t))

    def __repr__(self):
        return f"ObstacleWithKinematics({self.centroid[0]:.2f}, {self.centroid[1]:.2f})"
    
    def get_pose_at(self, t) -> tuple[float, float, float]:
        initial_state = self._initial_state
        return initial_state.x + initial_state.x_dot*t, initial_state.y + initial_state.y_dot*t, initial_state.psi_deg + initial_state.psi_dot_deg*t

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
    from nav_env.obstacles.collection import ObstacleWithKinematicsCollection, ObstacleCollection
    from nav_env.obstacles.obstacles import Obstacle
    from scipy.spatial.transform import Rotation as R

    o1 = ObstacleWithKinematics(pose_fn=lambda t: (t, t, 0), xy=[(0, 0), (2, 0), (2, 2), (0, 2)])
    o2 = ObstacleWithKinematics(pose_fn=lambda t: (-2*t, -t, 0), xy=[(0, 0), (2, 0), (3, 1), (2, 2), (0, 2)]).rotate(40).translate(10, 5)
    o3 = ObstacleWithKinematics(pose_fn=lambda t: (-1.2*t, -0.5*t, 0), xy=[(0, 0), (2, 0), (3, 1), (2, 2), (0, 2)]).rotate(40).translate(0, 5)
    o4 = ObstacleWithKinematics(pose_fn=lambda t: (2*t, 0.2*t, 0), xy=[(0, 0), (2, 0), (3, 1), (2, 2), (0, 2)]).rotate(40).translate(-3, -5)
    
    coll = ObstacleWithKinematicsCollection([o1, o2, o3, o4])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tmin, tmax = 0,  6
    
    for t in np.linspace(tmin, tmax, 50):
        ax = coll.plot3(t, ax=ax)

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
        for i, (p0, pf) in enumerate(zip(obs(tmin).buffer(1, join_style='mitre').get_xy_as_list(), obs(tmax).buffer(1, join_style='mitre').get_xy_as_list())):
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
    
    
    
    ax.set_zlim(0, np.max(t))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time')
    plt.show()

if __name__ == "__main__":
    # test_basic_obstacle()
    show_time_varying_obstacle_as_3d()