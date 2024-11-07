from abc import ABC, abstractmethod
from shapely import Polygon

class ObstacleBase(ABC):
    def __init__(self, shape:Polygon):
        self._shape = shape

    def __call__(self) -> Polygon:
        return self._shape
