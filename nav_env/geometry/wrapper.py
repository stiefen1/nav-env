from shapely import Polygon, LineString, Geometry, affinity, Point
import numpy as np, matplotlib.pyplot as plt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from copy import deepcopy
from typing import Any
import warnings

DEFAULT_IMAGE_SIZE = (100, 100)
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
        

    def plot(self, *args, ax=None, c=None, offset=None, **kwargs):
        """
        Plot the geometry.
        """

        if ax is None:
            _, ax = plt.subplots()

        if offset is None:
            offset = np.array([0., 0.])

        xy = self.translate(-offset[0], -offset[1]).xy
        ax.plot(*xy, *args, c=c, **kwargs)
        return ax
    
    def fill(self, *args, ax=None, c=None, **kwargs):
        """
        Fill the geometry.
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.fill(*self.xy, *args, c=c, **kwargs)
        return ax
    
    def scatter(self, *args, ax=None, **kwargs):
        """
        Scatter plot the geometry.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(*self.xy, *args, **kwargs)
        return ax

    def draw(self, screen:pygame.Surface, *args, scale=1, color=(255, 0, 0), **kwargs):
        """
        Draw the geometry for pygame.
        """
        screen_size = screen.get_size()
        transformed_points = [(scale*x + screen_size[0] // 2, screen_size[1] // 2 - scale*y) for x, y in self.get_xy_as_list()]
        pygame.draw.polygon(screen, color, transformed_points, *args, width=scale, **kwargs)


    def xy_as_list(self, xy) -> list:
        return list(zip(*xy))
    
    def get_xy_as_list(self) -> list:
        return self.xy_as_list(self.xy)

    def __call__(self) -> LineString:
        return self._geometry
    
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._geometry})"
    
    def __len__(self) -> int:
        return len(self.xy[0])
    
    @property
    def xy(self) -> tuple:
        if isinstance(self._geometry, Polygon):
            return self._geometry.exterior.coords.xy
        # if isinstance(self._geometry)
        # print(type(self._geometry))
        # print("XY:", self._geometry.xy)
        return self._geometry.xy
    
    @property
    def exterior(self) -> LineString:
        if isinstance(self._geometry, Polygon):
            return self._geometry.exterior
        elif isinstance(self._geometry, LineString):
            return self._geometry
        raise ValueError(f"Geometry must be a Polygon or LineString not {type(self._geometry).__name__}")
    
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
    
    def translate_inplace(self, x: float, y: float, **kwargs) -> None:
        self._geometry = affinity.translate(self._geometry, x, y, **kwargs)
    
    def rotate(self, angle: float, origin=None, **kwargs) -> "GeometryWrapper":
        """
        Rotation in degrees by default. For radians set use_radians=True
        """
        new = deepcopy(self)
        new._geometry = affinity.rotate(self._geometry, angle, origin=origin or self.centroid, **kwargs)
        return new
    
    def rotate_inplace(self, angle: float, origin=None, **kwargs) -> None:
        """
        Rotation in degrees by default. For radians set use_radians=True
        """
        self._geometry = affinity.rotate(self._geometry, angle, origin=origin or self.centroid, **kwargs)
    
    def rotate_and_translate(self, x:float, y:float, angle:float, origin=None, **kwargs) -> "GeometryWrapper":
        """
        Rotation in degrees by default. For radians set use_radians=True
        """
        new = deepcopy(self)
        new._geometry = affinity.translate(affinity.rotate(new._geometry, angle, origin=origin or self.centroid, **kwargs), x, y)
        return new
    
    def rotate_and_translate_inplace(self, x:float, y:float, angle:float, origin=None, **kwargs) -> None:
        """
        Rotation in degrees by default. For radians set use_radians=True
        """
        self._geometry = affinity.translate(affinity.rotate(self._geometry, angle, origin=origin or self.centroid, **kwargs), x, y)

    def translate_and_rotate(self, x:float, y:float, angle:float, origin=None, **kwargs) -> "GeometryWrapper":
        """
        Rotation in degrees by default. For radians set use_radians=True
        """
        new = deepcopy(self)
        new._geometry = affinity.rotate(affinity.translate(new._geometry, x, y), angle, origin=origin or (x, y), **kwargs)
        return new
    
    def translate_and_rotate_inplace(self, x:float, y:float, angle:float, origin=None, **kwargs) -> None:
        """
        Rotation in degrees by default. For radians set use_radians=True
        """
        self._geometry = affinity.rotate(affinity.translate(self._geometry, x, y), angle, origin=origin or (x, y), **kwargs)

    def center_inplace(self) -> None:
        self.translate_inplace(-self.centroid[0], -self.centroid[1])
    
    """
    Wrapper for shapely methods. Inheritance is impossible due to the way Shapely is implemented.
    """
    def interpolate(self, coord, *args, **kwargs) -> Point:
        return self._geometry.interpolate(coord, *args, **kwargs)
    
    def resample(self, n_wpts:int) -> "GeometryWrapper":
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
    
    def union(self, other, **kwargs) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = self._geometry.union(get_geometry_from_object(other), **kwargs)
        return new 
    
    def difference(self, other, **kwargs) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = self._geometry.difference(get_geometry_from_object(other), **kwargs)
        return new
    
    def buffer(self, distance: float, **kwargs) -> "GeometryWrapper":
        new = deepcopy(self)
        # print("0", new.centroid, self.centroid)
        
        try: # For MovingObstacle objects
            # print("#-------------------------Moving Obstacle------------------------------#")
            new._initial_geometry = new._initial_geometry.buffer(distance, **kwargs)
        except:
            # print("#-------------------------Static Obstacle------------------------------#")
            new.center_inplace()
            # print("1", new.centroid, self.centroid)
            new._geometry = new._geometry.buffer(distance, **kwargs) 
            new.center_inplace() # Useful because buffering might change centroid
            # print("2", new.centroid, self.centroid)

            new.translate_inplace(*self.centroid)
            # print("3", new.centroid, self.centroid)
        return new
    
    def simplify(self, tolerance: float, **kwargs) -> "GeometryWrapper":
        new = deepcopy(self)
        new._geometry = self._geometry.simplify(tolerance=tolerance, **kwargs)
        return new
    
    def simplify_inplace(self, tolerance: float, **kwargs) -> None:
        self._geometry.simplify(tolerance=tolerance, **kwargs)
    
    def distance(self, other, **kwargs) -> float:
        return self._geometry.distance(get_geometry_from_object(other), **kwargs)
    
    
def get_geometry_from_object(geometry: Geometry|GeometryWrapper|tuple) -> GeometryWrapper:
    if isinstance(geometry, (Polygon, LineString)):
        return geometry
    elif isinstance(geometry, tuple):
        return Point(*geometry)
    return geometry()