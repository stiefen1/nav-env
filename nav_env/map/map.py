from nav_env.obstacles.collection import ObstacleCollection
from nav_env.obstacles.obstacles import Obstacle
from seacharts.enc import ENC
import matplotlib.pyplot as plt
from shapely import intersects, Polygon, difference, intersection, MultiPolygon

class Map(ObstacleCollection):
    """
    A map object allows to convert ENC data into obstacles object that can be used for trajectory planning.
    """
    def __init__(self, path_to_config:str, center:tuple=None, size:tuple=None, depth:int=5):
        # Load using seacharts
        self.enc = ENC(path_to_config)       
        obstacles = self.get_obstacle_collection_in_window_from_enc(center=center, size=size, depth=depth)
        super().__init__(obstacles.as_list())

    def get_obstacle_collection_in_window_from_enc(self, center:tuple=None, size:tuple=None, depth:float=0.) -> ObstacleCollection:
        """
        If no size or center are provided, the obstacles will be collected according to the info available in seacharts config file.
        Specifying size and/or center allows the user to keep only obstacles from a specific region
        """
        
        lx_enc, ly_enc = self.enc.size
        x0_enc, y0_enc = self.enc.center
        window_complete = Polygon(((x0_enc-lx_enc/2, y0_enc-ly_enc/2), (x0_enc+lx_enc/2, y0_enc-ly_enc/2), (x0_enc+lx_enc/2, y0_enc+ly_enc/2), (x0_enc-lx_enc/2, y0_enc+ly_enc/2)))

        size_focus = size or self.enc.size
        center_focus = center or self.enc.center

        lx_focus, ly_focus = size_focus
        x0_focus, y0_focus = center_focus
        window_focus = Polygon(((x0_focus-lx_focus/2, y0_focus-ly_focus/2), (x0_focus+lx_focus/2, y0_focus-ly_focus/2), (x0_focus+lx_focus/2, y0_focus+ly_focus/2), (x0_focus-lx_focus/2, y0_focus+ly_focus/2)))

        obstacles = [] # Initialize list of obstacles representing the shore

        # Takes first depth_i that is greater or equal to depth
        for depth_i, seabed in zip(self.enc.seabed.keys(), self.enc.seabed.values()):
            if depth_i >= depth:
                list_of_polygons = list(seabed.geometry.geoms)
                for polygon in list_of_polygons:
                    multi_diff = intersection(difference(window_complete, polygon), window_focus) # We make sure that the resulting polygon is both part of enc and focus regions

                    # Difference can lead to multiple polygons. In such case we add them all to the obstacles collection
                    if isinstance(multi_diff, MultiPolygon):
                        for diff in multi_diff.geoms:
                            obstacles.append(Obstacle(polygon=diff, depth=depth_i))

                    # If difference is a single polygon, we just add it to the collection
                    elif isinstance(multi_diff, Polygon):
                        if multi_diff.area == window_focus.area: # For some reasons, window_focus is sometimes the result of it
                            continue
                        obstacles.append(Obstacle(polygon=multi_diff, depth=depth_i))
                break

        return ObstacleCollection(obstacles)

    def plot3(self, *args, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        for obs in self._obstacles:
            if obs.depth is not None:
                obs.plot3(-obs.depth, *args, ax=ax, **kwargs)
            else:
                UserWarning("plot3 called but obstacle's depth was not specified")

        return ax

def test() -> None:
    import sys, os, matplotlib.pyplot as plt
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.insert(0, root_path)

    config_path = os.path.join('examples', 'config', 'config_seacharts.yaml')
    shore = Map(config_path, depth=0)
    shore.plot3()
    
    # shore.plot()
    print(len(shore))
    plt.show()
    obs = shore.get_obstacle_collection_in_window_from_enc(center=(43150, 6958000), size=(1500, 850))
    obs.simplify(100)
    obs.plot()
    plt.show()
if __name__=="__main__":
    test()