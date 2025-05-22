from dataclasses import dataclass
import numpy as np
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
import pygame

"""
Ideal usage:

# pendulum with two states theta, theta_dot

s0 = States(theta=0.1, theta_dot=0.2)

"""

class BaseStateVector(ABC):
    def __init__(self, *args, **kwargs):
        if kwargs:
            self.__dict__.update(kwargs)
        else:
            for i, value in enumerate(args):
                self.__dict__[f'x{i+1}'] = value

    def __plot__(self, xy, keys, *args, ax=None, **kwargs):
        """
        Plot the states vector.
        """

        if ax is None:
            _, ax = plt.subplots()
        
        ax.quiver(*xy, self.__dict__[keys[0]], self.__dict__[keys[1]], *args, **kwargs)
        return ax
    
    def __draw__(self, screen:pygame.Surface, xy, keys, *args, color=(0, 255, 0), scale=1, unit_scale=1, **kwargs):
        """
        Draw the vector for pygame.
        """
        vx, vy = self.__dict__[keys[0]], self.__dict__[keys[1]]
        x, y = xy
        screen_size = screen.get_size()

        x = scale * x + screen_size[0] // 2
        y = screen_size[1] // 2 - scale * y

        x_tip = x + unit_scale * scale * vx
        y_tip = y - unit_scale * scale * vy

        pygame.draw.line(screen, color, (x, y), (x_tip, y_tip), *args, **kwargs)

    def __eq__(self, other:"BaseStateVector") -> tuple[bool]:
        list_of_bool = []
        for key in self.__dict__.keys():
            if self[key] == other[key]:
                list_of_bool.append(True)
            else:
                list_of_bool.append(False)
        return tuple(list_of_bool)
    
    def __ge__(self, other:"BaseStateVector") -> tuple[bool]:
        list_of_bool = []
        for key in self.__dict__.keys():
            if self[key] >= other[key]:
                list_of_bool.append(True)
            else:
                list_of_bool.append(False)
        return tuple(list_of_bool)
    
    def __gt__(self, other:"BaseStateVector") -> tuple[bool]:
        list_of_bool = []
        for key in self.__dict__.keys():
            if self[key] > other[key]:
                list_of_bool.append(True)
            else:
                list_of_bool.append(False)
        return tuple(list_of_bool)
    
    def __ne__(self, other:"BaseStateVector") -> tuple[bool]:
        list_of_bool = []
        for key in self.__dict__.keys():
            if self[key] != other[key]:
                list_of_bool.append(True)
            else:
                list_of_bool.append(False)
        return tuple(list_of_bool)
    
    def __neg__(self):
        new_dict = {}
        for key, val in zip(self.keys, self.values):
            new_dict.update({key: -val})
        return type(self).__call__(**new_dict)
    
    def __len__(self) -> int:
        return len(self.keys)

    def __mul__wrapper__(self, scalar:float, output_type):
        new_values = tuple([value * scalar for value in self.__dict__.values()])
        new_dict = dict(zip(self.__dict__.keys(), new_values))
        return output_type.__call__(**new_dict)
    
    def __add__wrapper__(self, other:"BaseStateVector", keys, output_type):
        new_values = tuple([x + y for x, y in zip(self.__dict__.values(), other.__dict__.values())])
        new_dict = dict(zip(keys, new_values))
        return output_type.__call__(**new_dict)

    def __add__(self, other:"BaseStateVector"):
        return self.__add__wrapper__(other, other.keys, type(self))
    
    def __sub__(self, other:"BaseStateVector"):
        return self.__add__(-1 * other)

    def __mul__(self, scalar:float):
        return self.__mul__wrapper__(scalar, type(self))

    def __rmul__(self, dt:float):
        return self.__mul__(dt)
    
    def __truediv__(self, scalar:float):
        return self.__mul__(1/scalar)
    
    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__})"
    
    def __getitem__(self, index):
        if isinstance(index, str):
            if index not in self.__dict__:
                raise KeyError(f"States variable '{index}' not in {list(self.__dict__.keys())}")
            return self.__dict__[index]
        elif isinstance(index, int):
            if index >= len(self.__dict__):
                raise IndexError(f"Index {index} out of range for {list(self.__dict__.keys())}")
            return list(self.__dict__.values())[index]
        else:
            raise TypeError(f"Index must be a string or an integer not {type(index).__name__}")

    def __setitem__(self, index, value):
        if isinstance(index, str):
            if index not in self.__dict__:
                raise KeyError(f"States variable '{index}' not in {list(self.__dict__.keys())}")
            self.__dict__[index] = value
        elif isinstance(index, int):
            if index >= len(self.__dict__):
                raise IndexError(f"Index {index} out of range for {list(self.__dict__.keys())}")
            self.__dict__[list(self.__dict__.keys())[index]] = value
        else:
            raise TypeError(f"Index must be a string or an integer not {type(index).__name__}")
        
    @property    
    def keys(self) -> list:
        return list(self.__dict__.keys())
    
    @property
    def values(self) -> list:
        return list(self.__dict__.values())
    
    @property
    def dim(self) -> int:
        return len(self.__dict__)


class States(BaseStateVector): # We use states space representation \dot{x} = f(x, u)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __add__(self, other):
        if isinstance(other, DeltaStates):
            return self.__add__wrapper__(other, self.keys, type(self))
        raise TypeError(f"Cannot add {type(self).__name__} object with {type(other).__name__}")
    
    def __sub__(self, other):
        if isinstance(other, DeltaStates):
            return self.__add__wrapper__(-1 * other, self.keys, type(self))
        elif isinstance(other, BaseStateVector):
            return self.__add__wrapper__(-1 * other, self.keys, DeltaStates)
        raise TypeError(f"Cannot substract {type(self).__name__} object with {type(other).__name__}")

class DeltaStates(BaseStateVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __add__(self, other):
        """
        Add a delta states to a states or another delta states.
        """
        if isinstance(other, BaseStateVector):
            return self.__add__wrapper__(other, self.keys, type(other))
        raise TypeError(f"Cannot add {type(self).__name__} object with {type(other).__name__}")
    
    def __sub__(self, other):
        return self.__add__(-1 * other)

class TimeDerivatives(BaseStateVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __mul__(self, scalar:float):
        return self.__mul__wrapper__(scalar, DeltaStates)

def test():
    x = States(x=2, y=3, yaw=4)
    
    print(len(x.__dict__), x.x, x.y, x.yaw)
    x.y = -10.
    print(len(x.__dict__), x.x, x.y, x.yaw)
    print(x)

    dx = DeltaStates(0, 1, 2)
    x0 = x + dx # ATTENTION CA NE MARCHE QUE DANS UN SENS PAS L'AUTRE
    print(x0)

    dxdt = TimeDerivatives(1, 2, 3)
    print(dxdt)
    dx = dxdt * 0.1
    print(dx)
    y = x0 + dx
    print(y)
    z = dxdt * 0.1 + dx + x0
    print(z - dx)

    print(x['x'], x[0]/2, x.x/4)
    x['x'] = 10
    print(x['x'], x[0]/2, x.x/4)
    x[0] = 20
    print(x['x'], x[0]/2, x.x/4)
    x.x = 30
    print(x['x'], x[0]/2, x.x/4)

    x = States(x=2, y=3, yaw=4)
    y = States(x=2, y=4, yaw=2)
    print(x==y)
    print(x<=y)
    print(x>y)
    print(x>=y)
    print(x!=y)

if __name__ == "__main__":
    test()