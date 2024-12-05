from nav_env.control.states import States, TimeDerivatives
from math import pi

class ShipStates3(States):
    def __init__(self, x:float=0., y:float=0., psi_deg:float=0., x_dot:float=0., y_dot:float=0., psi_dot_deg:float=0.):
        super().__init__(x=x, y=y, psi_deg=psi_deg, x_dot=x_dot, y_dot=y_dot, psi_dot_deg=psi_dot_deg)

    def plot(self, *args, ax=None, **kwargs):
        """
        Plot the state vector.
        """
        return self.__plot__(self.xy, ['x_dot', 'y_dot'], *args, ax=ax, **kwargs)

    @property
    def x(self) -> float:
        return self['x']
    
    @x.setter
    def x(self, value:float):
        self['x'] = value

    @property
    def y(self) -> float:
        return self['y']
    
    @y.setter
    def y(self, value:float):
        self['y'] = value

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y
    
    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.psi_rad

    ####### psi is stored as degrees #######
    @property
    def psi_rad(self) -> float:
        return self['psi_deg'] * pi / 180  # Convert to radians
    
    @psi_rad.setter
    def psi_rad(self, value:float):
        self['psi_deg'] = value * 180 / pi # Convert to degrees to store

    @property
    def psi_deg(self) -> float:
        return self['psi_deg']
    
    @psi_deg.setter
    def psi_deg(self, value:float):
        self['psi_deg'] = value

    @property
    def x_dot(self) -> float:
        return self['x_dot']
    
    @x_dot.setter
    def x_dot(self, value:float):
        self['x_dot'] = value

    @property
    def y_dot(self) -> float:
        return self['y_dot']
    
    @y_dot.setter
    def y_dot(self, value:float):
        self['y_dot'] = value

    @property
    def xy_dot(self) -> tuple[float, float]:
        return self.x_dot, self.y_dot
    
    @property
    def vel(self) -> tuple[float, float, float]:
        return self.x_dot, self.y_dot, self.psi_dot_rad

    @property
    def psi_dot_deg(self) -> float:
        return self['psi_dot_deg']
    
    @psi_dot_deg.setter
    def psi_dot_deg(self, value:float):
        self['psi_dot_deg'] = value

    @property
    def psi_dot_rad(self) -> float:
        return self['psi_dot_deg'] * pi / 180  # Convert to radians
    
    @psi_dot_rad.setter
    def psi_dot_rad(self, value:float):
        self['psi_dot_deg'] = value * 180 / pi # Convert to degrees to store

class ShipTimeDerivatives3(TimeDerivatives):
    def __init__(self, x_dot:float=0., y_dot:float=0., psi_dot_deg:float=0., x_dot_dot:float=0., y_dot_dot:float=0., psi_dot_dot:float=0.):
        super().__init__(x_dot=x_dot, y_dot=y_dot, psi_dot_deg=psi_dot_deg, x_dot_dot=x_dot_dot, y_dot_dot=y_dot_dot, psi_dot_dot=psi_dot_dot)

    def plot_acc(self, xy, *args, ax=None, **kwargs):
        """
        Plot the state vector.
        """
        return self.__plot__(xy, ['x_dot_dot', 'y_dot_dot'], *args, ax=ax, **kwargs)
    
    def plot_vel(self, xy, *args, ax=None, **kwargs):
        """
        Plot the state vector.
        """
        return self.__plot__(xy, ['x_dot', 'y_dot'], *args, ax=ax, **kwargs)

    @property
    def x_dot(self) -> float:
        return self['x_dot']
    
    @x_dot.setter
    def x_dot(self, value:float):
        self['x_dot'] = value

    @property
    def y_dot(self) -> float:
        return self['y_dot']
    
    @y_dot.setter
    def y_dot(self, value:float):
        self['y_dot'] = value

    @property
    def psi_dot_deg(self) -> float:
        return self['psi_dot_deg']
    
    @psi_dot_deg.setter
    def psi_dot_deg(self, value:float):
        self['psi_dot_deg'] = value

    @property
    def x_dot_dot(self) -> float:
        return self['x_dot_dot']
    
    @x_dot_dot.setter
    def x_dot_dot(self, value:float):
        self['x_dot_dot'] = value

    @property
    def y_dot_dot(self) -> float:
        return self['y_dot_dot']
    
    @y_dot_dot.setter
    def y_dot_dot(self, value:float):
        self['y_dot_dot'] = value

    @property
    def psi_dot_dot(self) -> float:
        return self['psi_dot_dot']
    
    @psi_dot_dot.setter
    def psi_dot_dot(self, value:float):
        self['psi_dot_dot'] = value

def test():
    states = ShipStates3(x=1., y=2., psi_deg=3., x_dot=4., y_dot=5., psi_dot_deg=6.)
    print(states)
    states = 3 * states - states / 2
    print(states)

if __name__ == "__main__":
    test()