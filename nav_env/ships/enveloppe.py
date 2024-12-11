from nav_env.obstacles.obstacles import Obstacle
import os, pathlib

PATH_TO_DEFAULT_IMG = os.path.join(pathlib.Path(__file__).parent.parent, "ships", "ship.png")

DEFAULT_SHIP_LENGTH = 10.
DEFAULT_SHIP_WIDTH = 4.
DEFAULT_SHIP_RATIO = 10/15 # = length of the lower rectangle / tot length
DEFAULT_SHIP_SPEED = (1., 1., 0.)
DEFAULT_SHIP_POSITION = (0., 0., 0.)

def get_ship_enveloppe(length, width, ratio):
    # Make the centroid of this enveloppe placed at (0, 0)

    # Upper triangle area:
    triangle_height = (1-ratio) * length
    triangle_area = width * triangle_height / 2
    # Upper triange centroid y:
    triangle_centroid_y = triangle_height / 3
    # Lower rectangle area:
    rectangle_height = ratio * length
    rectangle_area = rectangle_height * width
    # Lower rectangle centroid y:
    rectangle_centroid_y = -rectangle_height / 2
    # Overall centroid
    centroid_y = (triangle_area * triangle_centroid_y + rectangle_area * rectangle_centroid_y) / (triangle_area + rectangle_area)

    enveloppe = [
        [0, triangle_height - centroid_y],
        [-width/2, 0 - centroid_y],
        [-width/2, -rectangle_height-centroid_y],
        [width/2, -rectangle_height-centroid_y],
        [width/2, 0 - centroid_y]
    ]
    return enveloppe

class ShipEnveloppe(Obstacle):
    def __init__(self,
                 length: float=None,
                 width: float=None,
                 ratio: float=None,
                 img:str=None
                 ):
        
        self._length = length or DEFAULT_SHIP_LENGTH
        self._width = width or DEFAULT_SHIP_WIDTH
        self._ratio = ratio or DEFAULT_SHIP_RATIO
        self._img = img or PATH_TO_DEFAULT_IMG

        xy = get_ship_enveloppe(self._length, self._width, self._ratio)

        super().__init__(xy=xy, img=img)

    def plot(self, ax=None, c='r', alpha=1, **kwargs):
        return super().plot(ax=ax, c=c, alpha=alpha, **kwargs)
    
    @property
    def length(self) -> float: 
        return self._length
    
    @property
    def width(self) -> float:
        return self._width
    
    @property
    def ratio(self) -> float:
        return self._ratio