from shapely import Polygon, LineString, Geometry, affinity
import numpy as np, matplotlib.pyplot as plt
import pygame
from copy import deepcopy


class GeometryWrapper:
    def __init__(self, xy: list=None, polygon: Geometry=None, geometry_type: type=Geometry):
        if polygon is None and xy is not None:
            self._geometry = geometry_type(xy)
            self._geometry_type = geometry_type
        elif polygon is not None:
            self._geometry = polygon
            self._geometry_type = type(polygon)
        else:
            raise ValueError("Either xy or polygon must be provided.")

    def plot(self, ax=None, **kwargs):
        """
        Plot the geometry.
        """

        if ax is None:
            _, ax = plt.subplots()
        
        ax.plot(*self.xy, **kwargs)
        return ax
    
    def scatter(self, ax=None, **kwargs):
        """
        Scatter plot the geometry.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(*self.xy, **kwargs)
        return ax
    
    def draw(self, ax=None, **kwargs):
        """
        Draw the geometry for pygame.
        """
        if ax is None:
            ax = pygame.display.get_surface()
        pygame.draw.polygon(ax, (0, 0, 0), self.xy, **kwargs)
        return ax

    def __call__(self) -> LineString:
        return self._geometry
    
    def __repr__(self) -> str:
        return f"GeometryWrapper({self._geometry})"
    
    @property
    def xy(self) -> tuple:
        if isinstance(self._geometry, Polygon):
            return self._geometry.exterior.coords.xy
        return self._geometry.xy
    
    @property
    def centroid(self) -> tuple:
        return self._geometry.centroid.coords[0]

    @centroid.setter
    def centroid(self, value: tuple) -> None:
        self._geometry = affinity.translate(self._geometry, value[0] - self.centroid[0], value[1] - self.centroid[1])

    def translate(self, x: float, y: float, **kwargs) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = affinity.translate(self._geometry, x, y, **kwargs)
        return new
    
    def rotate(self, angle: float, **kwargs) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = affinity.rotate(self._geometry, angle, origin=self.centroid, **kwargs)
        return new
    
    """
    Wrapper for shapely methods. Inheritance is impossible due to the way Shapely is implemented.
    """
    def interpolate(self, n_wpts:int) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = self._geometry_type(self._geometry.interpolate(np.linspace(0, 1, n_wpts), normalized=True))
        return new
    
    def crosses(self, other, **kwargs) -> bool:
        return self._geometry.crosses(get_geometry_from_object(other), **kwargs)
    
    def touches(self, other, **kwargs) -> bool:
        return self._geometry.touches(get_geometry_from_object(other), **kwargs)
    
    def intersects(self, other, **kwargs) -> bool:
        return self._geometry.intersects(get_geometry_from_object(other), **kwargs)
    
    def contains(self, other, **kwargs) -> bool:
        return self._geometry.contains(get_geometry_from_object(other), **kwargs)
    
    def within(self, other, **kwargs) -> bool:
        return self._geometry.within(get_geometry_from_object(other), **kwargs)
    
    def intersection(self, other, **kwargs) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = self._geometry.intersection(get_geometry_from_object(other), **kwargs)
        return new
    
    def convex_hull(self) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = self._geometry.convex_hull
        return new
    
    def convex_hull_of_union(self, other, **kwargs) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = self._geometry.union(get_geometry_from_object(other), **kwargs).convex_hull
        return new
    
    def difference(self, other, **kwargs) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = self._geometry.difference(get_geometry_from_object(other), **kwargs)
        return new
    
    def buffer(self, distance: float, **kwargs) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = self._geometry.buffer(distance, **kwargs)
        return new
    
    def simplify(self, tolerance: float, **kwargs) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = self._geometry.simplify(tolerance, **kwargs)
        return new
    
    def distance(self, other, **kwargs) -> float:
        return self._geometry.distance(get_geometry_from_object(other), **kwargs)
    
    
def get_geometry_from_object(geometry: Geometry|GeometryWrapper) -> GeometryWrapper:
    if isinstance(geometry, (Polygon, LineString)):
        return geometry
    return geometry()