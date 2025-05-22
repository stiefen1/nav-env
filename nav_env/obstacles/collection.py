from nav_env.obstacles.obstacles import Obstacle, MovingObstacle
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from shapely import intersection, Polygon, MultiPolygon

class ObstacleCollection:
    def __init__(self, obstacles: list[Obstacle] = []):
        assert isinstance(obstacles, list), f"Expected list got {type(obstacles).__name__}"
        self._obstacles = obstacles

    def append(self, obstacle: Obstacle):
        assert isinstance(obstacle, Obstacle), f"Obstacle must be an instance of Obstacle not {type(obstacle)}"
        self._obstacles.append(obstacle)

    def remove(self, obstacle: Obstacle):
        self._obstacles.remove(obstacle)

    def distance_to_obstacles(self, x:float, y:float) -> float:
        """
        Get the distance to the obstacle at a given position.
        """
        return [obs.distance_to_obstacle(x, y) for obs in self._obstacles]
    
    def group_obstacles_closer_than(self, min_distance:float, recursive:bool=False) -> None:
        """
        Combine obstacles that are closer (between them) than a given distance using their convex hull.
        """
        obstacles = self._obstacles
        graph = get_graph_of_connected_obstacles(self, min_distance)
        new_obstacles = get_convex_hull_of_union_of_connected_obstacles_as_list(graph, obstacles)

        self._obstacles = new_obstacles

        if recursive and some_obstacles_are_too_close(new_obstacles, min_distance):
            self.group_obstacles_closer_than(min_distance, recursive=True)

    def get_group_of_obstacles_closer_than(self, min_distance:float, recursive:bool=False) -> "ObstacleCollection":
        """
        Get a new collection of obstacles where obstacles closer than a given distance are combined.
        """
        new = deepcopy(self)
        new.group_obstacles_closer_than(min_distance, recursive=recursive)
        return new
    
    def get_group_of_obstacles_in_window(self, center:tuple, size:tuple) -> "ObstacleCollection":
        x0, y0 = center
        lx, ly = size
        window = Polygon(((x0-lx/2, y0-ly/2), (x0+lx/2, y0-ly/2), (x0+lx/2, y0+ly/2), (x0-lx/2, y0+ly/2)))
        obstacles = []
        for obs in self._obstacles:
            inter = intersection(window, obs._geometry)
            if isinstance(inter, MultiPolygon): # Might happen for non-convex obstacles
                for polygon in list(inter.geoms):
                    if polygon.area > 0:
                        obstacles.append(Obstacle(polygon=polygon, depth=obs.depth))
            elif isinstance(inter, Polygon):
                if inter.area > 0:
                    obstacles.append(Obstacle(polygon=inter, depth=obs.depth))
        return ObstacleCollection(obstacles)
    
    def group_intersecting_obstacles(self) -> "ObstacleCollection":
        """
        Combine obstacles whose intersection (between them) is not null --> I BELIEVE IT IS NOT WORKING PROPERLY 
        """
        obstacles = self._obstacles
        graph = get_graph_of_intersecting_obstacles(self)
        new_obstacles = get_union_of_connected_obstacles_as_list(graph, obstacles)

        self._obstacles = new_obstacles

        # if recursive and some_obstacles_are_too_close(new_obstacles, min_distance):
        #     self.group_obstacles_closer_than(min_distance, recursive=True)

    def get_collection_of_convex_hull(self) -> "ObstacleCollection":
        new_obstacles = []
        for obs in self._obstacles:
            # obs.plot()
            # plt.show()
            new_obs = Obstacle(obs.convex_hull().get_xy_as_list())
            new_obstacles.append(new_obs)
        return ObstacleCollection(new_obstacles)

    def buffer(self, distance:float, **kwargs) -> "ObstacleCollection":
        """
        Buffer the obstacles.
        """
        return ObstacleCollection([obs.buffer(distance, **kwargs) for obs in self._obstacles])
    
    def intersection(self, other, **kwargs) -> list:
        """
        Get the intersection of the obstacles with another geometry.
        """
        return [obs.intersection(other, **kwargs) for obs in self._obstacles]
    
    def translate(self, x:float, y:float) -> "ObstacleCollection":
        """
        Translate the obstacles.
        """
        return ObstacleCollection([obs.translate(x, y) for obs in self._obstacles])
    
    def simplify(self, tolerance: float, **kwargs) -> None:
        for obs in self._obstacles:
            obs.simplify_inplace(tolerance=tolerance, **kwargs)

    def as_list(self) -> list:
        return self._obstacles

    def plot(self, *args, ax=None, text=False, **kwargs):
        """
        Plot the obstacles.
        """

        if ax is None:
            _, ax = plt.subplots()
        
        for i, obs in enumerate(self._obstacles):
            obs.plot(*args, ax=ax, **kwargs)
            if text:
                ax.text(*obs.centroid, f'{i}: {str(obs)}')
        return ax
    
    def fill(self, *args, ax=None, text=False, **kwargs):
        """
        Plot the obstacles.
        """

        if ax is None:
            _, ax = plt.subplots()
        
        for i, obs in enumerate(self._obstacles):
            obs.fill(*args, ax=ax, **kwargs)
            if text:
                ax.text(*obs.centroid, f'{i}: {str(obs)}')
        return ax

    def plot3(self, z:float, *args, ax=None, **kwargs):
        """
        Plot the obstacle in 3D.
        """
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        for obs in self._obstacles:
            obs.plot3(z, *args, ax=ax, **kwargs)

        return ax

    def draw(self, screen, scale=1, **kwargs):
        """
        Draw the obstacles.
        """
        for obs in self._obstacles:
            obs.draw(screen, scale=scale)

    def __getitem__(self, index: int) -> Obstacle:
        return self._obstacles[index]
    
    def __len__(self) -> int:
        return len(self._obstacles)

    def __repr__(self):
        return f"ObstacleCollection({len(self._obstacles)} obstacles)"

    def __iter__(self):
        for obs in self._obstacles:
            yield obs


class MovingObstacleCollection:
    def __init__(self, obstacles: list[MovingObstacle] = []):
        assert isinstance(obstacles, list), f"Expected list got {type(obstacles).__name__}"
        self._obstacles = obstacles

    def append(self, obstacle: MovingObstacle):
        assert isinstance(obstacle, MovingObstacle), f"Obstacle must be an instance of Obstacle not {type(obstacle)}"
        self._obstacles.append(obstacle)

    def remove(self, obstacle: MovingObstacle):
        self._obstacles.remove(obstacle)

    def plot3(self, t:float|tuple[float, float], ax=None, **kwargs):
        """
        Plot the obstacles in 3D.
        """
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        for obs in self._obstacles:
            obs.plot3(t, ax=ax, **kwargs)
        return ax
    
    def plot3_polyhedra_with_uncertainties(self, t:float, intervals:list[dict] | dict, *args, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        if not(isinstance(intervals, list)):
            intervals = len(self._obstacles) * [intervals]

        assert self.__len__() == len(intervals), f"You must specify either one or as many intervals as you have obstacles. Got {len(intervals)} intervals instead of {self.__len__()}."            

        for obs, interval in zip(self._obstacles, intervals):
            obs.plot3_polyhedron_with_uncertainties(t, interval, *args, ax=ax, **kwargs)
        return ax
    
    def quiver_speed(self, ax=None, **kwargs):
        """
        Plot the speed of the obstacles.
        """
        if ax is None:
            _, ax = plt.subplots()

        for obs in self._obstacles:
            obs.quiver_speed(ax=ax, **kwargs)
        return ax
    
    def plot(self, ax=None, domain:bool=False, **kwargs):
        """
        Plot the obstacles.
        """
        if ax is None:
            _, ax = plt.subplots()

        for obs in self._obstacles:
            obs.plot(ax=ax, domain=domain, **kwargs)
        return ax
    
    def draw(self, screen, scale=1, **kwargs):
        """
        Draw the obstacles.
        """
        for obs in self._obstacles:
            obs.draw(screen, scale=scale, **kwargs)
    
    def step(self) -> None:
        """
        Step the obstacles.
        """
        for obs in self._obstacles:
            obs.step()

    def reset(self) -> None:
        """
        Reset the obstacles.
        """
        for obs in self._obstacles:
            obs.reset()

    def set_integration_step(self, dt:float) -> None:
        """
        Set the integration step for all obstacles.
        """
        for obs in self._obstacles:
            obs.dt = dt

    def __call__(self, t:float) -> ObstacleCollection:
        return ObstacleCollection([obs(t) for obs in self._obstacles])

    def __getitem__(self, index: int) -> Obstacle:
        return self._obstacles[index]
    
    def __len__(self) -> int:
        return len(self._obstacles)

    def __repr__(self):
        return f"ObstacleCollection({len(self._obstacles)} obstacles)"

    def __iter__(self):
        for obs in self._obstacles:
            yield obs


def get_graph_of_intersecting_obstacles(obstacles: ObstacleCollection) -> nx.Graph:
    """
    Get a graph of intersecting obstacles.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(obstacles)))
    for i in range(1, len(obstacles)):
        obs_i = obstacles[i]
        for j in range(i):
            obs_j = obstacles[j]
            if obs_j.intersects(obs_i):
                graph.add_edge(i, j)
            # d = obs_i.distance(obs_j)
            # if d < min_distance:
            #     graph.add_edge(i, j)
    return graph

def get_union_of_connected_obstacles_as_list(graph: nx.Graph, obstacles: list | ObstacleCollection) -> list[set]:
    groups = list(nx.connected_components(graph))
    # combine obstacles in each group
    new_obstacles = []
    for group in groups:
        group_list = list(group)
        new_obstacle = obstacles[group_list[0]]
        for i in range(1, len(group_list)):
            obs_i = obstacles[group_list[i]]
            print("Intersection? ", obs_i.intersects(new_obstacle))
            print(type(new_obstacle.union(obs_i)()))
            ax = new_obstacle.plot()
            obs_i.plot(ax=ax)
            plt.show()

            new_obstacle = Obstacle(new_obstacle.union(obs_i).get_xy_as_list())
        new_obstacles.append(new_obstacle)
    return new_obstacles

def get_graph_of_connected_obstacles(obstacles: ObstacleCollection, min_distance:float) -> nx.Graph:
    """
    Get a graph of connected obstacles.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(obstacles)))
    for i in range(1, len(obstacles)):
        obs_i = obstacles[i]
        for j in range(i):
            obs_j = obstacles[j]
            d = obs_i.distance(obs_j)
            if d < min_distance:
                graph.add_edge(i, j)
    return graph

def get_convex_hull_of_union_of_connected_obstacles_as_list(graph: nx.Graph, obstacles) -> list[set]:
    groups = list(nx.connected_components(graph))
    # combine obstacles in each group
    new_obstacles = []
    for group in groups:
        group_list = list(group)
        new_obstacle = obstacles[group_list[0]]
        for i in range(1, len(group_list)):
            obs_i = obstacles[group_list[i]]
            new_obstacle = Obstacle(new_obstacle.convex_hull_of_union(obs_i)())
        new_obstacles.append(new_obstacle)
    return new_obstacles

def some_obstacles_are_too_close(obstacles:list[Obstacle], min_distance:float) -> bool:
    # check if there are still intersections
    for obs_i in obstacles:
        for obs_j in obstacles:
            if obs_i != obs_j:
                if obs_i.distance(obs_j) < min_distance:
                    return True
    return False

def test_collection():
    from nav_env.obstacles.obstacles import Circle
    from nav_env.obstacles.obstacles import Obstacle
    import numpy as np

    Ncircle = 30 # Number of obstacles to generate
    Npoly = 30 # Number of obstacles to generate
    lim = ((-10, -10), (10, 10))
    xmin, ymin = lim[0]
    xmax, ymax = lim[1]
    rmin, rmax = 0.1, 1
    obsmin, obsmax = -1.5, 1.5

    obstacles = ObstacleCollection()
    for _ in range(Ncircle):
        c = Circle(np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax), np.random.uniform(rmin, rmax))
        obstacles.append(c)

    for _ in range(Npoly):
        xy = np.random.uniform(obsmin, obsmax, (8, 2))
        o = Obstacle(xy=xy).convex_hull().translate(np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))
        obstacles.append(o)

    obstacles.plot()
    obstacles.group_obstacles_closer_than(0.5)
    obstacles.plot()
    obstacles.group_obstacles_closer_than(0.5, recursive=True)
    obstacles.plot(text=True)
    plt.show()

def test_time_varying_collection():
    import time
    from nav_env.ships.states import States3
    o1 = MovingObstacle(pose_fn=lambda t: States3(t, 10-t, t*10), xy=[(0, 0), (2, 0), (2, 2), (0, 2)])
    o2 = MovingObstacle(pose_fn=lambda t: States3(t, t, t*20), xy=[(0, 0), (2, 0), (2, 2), (0, 2)])
    coll = MovingObstacleCollection([o1, o2])
    ax = coll(0).plot()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.pause(1)

    dt:float = 0.02
    start = time.time()
    while True:
        start_loop = time.time()
        ax.cla()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        t = time.time() - start
        coll(t).get_group_of_obstacles_closer_than(1.).plot(ax=ax)
        
        if (t % 1) > (1-dt):
            inter = time.time()
            print(t, inter - start)
        
        end = time.time()
        if end - start_loop < dt:
            plt.pause(dt - (end - start_loop))
        else:
            print("Loop took too long")
            

        if t > 10:
            break

if __name__ == "__main__":
    # test_collection()
    test_time_varying_collection()


    