from dataclasses import dataclass
import numpy as np
from abc import abstractmethod, ABC


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
                raise KeyError(f"State variable '{index}' not in {list(self.__dict__.keys())}")
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
                raise KeyError(f"State variable '{index}' not in {list(self.__dict__.keys())}")
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


class States(BaseStateVector): # We use state space representation \dot{x} = f(x, u)
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
        Add a delta state to a state or another delta state.
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

if __name__ == "__main__":
    test()