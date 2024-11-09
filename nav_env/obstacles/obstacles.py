from shapely import Polygon, Point
from nav_env.geometry import GeometryWrapper

class Obstacle(GeometryWrapper):
    def __init__(self, xy: list=None, polygon: Polygon=None, geometry_type: type=Polygon):
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
    
def test():
    from matplotlib import pyplot as plt
    c = Circle(0, 0, 1)
    o = Obstacle(xy=[(0, 0), (2, 0), (2, 2), (0, 2)]).translate(0, -0.5).rotate(45)
    c.center = (0, -1)
    c.radius = 2
    ax = o.plot()
    c.plot(ax=ax)
    c.difference(o).plot()
    c.convex_hull_of_union(o).plot()
    plt.show()

if __name__ == "__main__":
    test()
