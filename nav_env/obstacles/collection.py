from nav_env.obstacles.obstacles import Obstacle, TimeVaryingObstacle
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy

class ObstacleCollection:
    def __init__(self, obstacles: list[Obstacle] = []):
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
        new_obstacles = get_union_of_connected_obstacles_as_list(graph, obstacles)

        self._obstacles = new_obstacles

        if recursive and some_obstacles_are_too_close(new_obstacles, min_distance):
            self.group_obstacles_closer_than(min_distance, recursive=True)

    def get_group_of_obstacles_closer_than(self, min_distance:float) -> "ObstacleCollection":
        """
        Get a new collection of obstacles where obstacles closer than a given distance are combined.
        """
        new = deepcopy(self)
        new.group_obstacles_closer_than(min_distance)
        return new

    def buffer(self, distance:float, **kwargs) -> None:
        """
        Buffer the obstacles.
        """
        return ObstacleCollection([obs.buffer(distance, **kwargs) for obs in self._obstacles])

    def plot(self, ax=None, text=False, **kwargs):
        """
        Plot the obstacles.
        """

        if ax is None:
            _, ax = plt.subplots()
        
        for i, obs in enumerate(self._obstacles):
            obs.plot(ax=ax, **kwargs)
            if text:
                ax.text(*obs.centroid, f'{i}: {str(obs)}')
        return ax
    
    def draw(self, screen):
        """
        Draw the obstacles.
        """
        for obs in self._obstacles:
            obs.draw(screen)

    def __getitem__(self, index: int) -> Obstacle:
        return self._obstacles[index]
    
    def __len__(self) -> int:
        return len(self._obstacles)

    def __repr__(self):
        return f"ObstacleCollection({len(self._obstacles)} obstacles)"

    def __iter__(self):
        for obs in self._obstacles:
            yield obs


class TimeVaryingObstacleCollection:
    def __init__(self, obstacles: list[TimeVaryingObstacle] = []):
        self._obstacles = obstacles

    def append(self, obstacle: TimeVaryingObstacle):
        assert isinstance(obstacle, TimeVaryingObstacle), f"Obstacle must be an instance of Obstacle not {type(obstacle)}"
        self._obstacles.append(obstacle)

    def remove(self, obstacle: TimeVaryingObstacle):
        self._obstacles.remove(obstacle)

    def plot3(self, t:float, ax=None, **kwargs):
        """
        Plot the obstacles in 3D.
        """
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        for obs in self._obstacles:
            obs.plot3(t, ax=ax, **kwargs)
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

def get_union_of_connected_obstacles_as_list(graph: nx.Graph, obstacles) -> list[set]:
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
    o1 = TimeVaryingObstacle(lambda t: (t, -t, t*10), xy=[(0, 0), (2, 0), (2, 2), (0, 2)]).rotate(45).translate(0., 9.)
    o2 = TimeVaryingObstacle(lambda t: (t, t, t*20), xy=[(0, 0), (2, 0), (2, 2), (0, 2)]).rotate(45).translate(0., 0.)
    coll = TimeVaryingObstacleCollection([o1, o2])
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
            break

        if t > 10:
            break

if __name__ == "__main__":
    # test_collection()
    test_time_varying_collection()


    