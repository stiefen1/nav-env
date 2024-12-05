from nav_env.geometry.wrapper import GeometryWrapper
from shapely.geometry import Polygon, LineString

class Line(GeometryWrapper):
    def __init__(self, xy: list=None, polygon: LineString=None):
            super().__init__(xy=xy, polygon=polygon, geometry_type=LineString)

def test():
    import matplotlib.pyplot as plt
    from nav_env.obstacles.obstacles import Circle

    c = Circle(0, 0, 3.2)
    xy = [(0, 0, 0), (1, 1, 1), (1.5, 3, 2), (2, 2, 3)]
    line = Line(xy=xy)
    ax = line.plot()
    c.plot(ax=ax)
    print(line.interpolate((0, 1), normalized=True))
    print(line.intersection(c.exterior))
    plt.show()

if __name__ == "__main__":
    test()