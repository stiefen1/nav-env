from nav_env.control.states import BaseStateVector
import matplotlib.pyplot as plt

class Command(BaseStateVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def __add__(self, other):
    #     self.__add__wrapper__(other, other.keys, type(self))

class GeneralizedForces(Command):
    def __init__(self, f_x:float=0., f_y:float=0., f_z:float=0., tau_x:float=0., tau_y:float=0., tau_z:float=0.):
        super().__init__(f_x=f_x, f_y=f_y, f_z=f_z, tau_x=tau_x, tau_y=tau_y, tau_z=tau_z)   

    def plot(self, xy, *args, ax=None, **kwargs):
        """
        Plot the generalized forces.
        """
        return self.__plot__(xy, ['f_x', 'f_y'], *args, ax=ax, **kwargs)
    
    def draw(self, screen, xy, *args, scale=1, unit_scale=1, **kwargs):
        """
        Draw the vector for pygame.
        """
        self.__draw__(screen, xy, ['f_x', 'f_y'], *args, scale=scale, unit_scale=unit_scale, **kwargs)

    @property
    def f_x(self) -> float:
        return self['f_x']
    
    @f_x.setter
    def f_x(self, value:float):
        self['f_x'] = value

    @property
    def f_y(self) -> float:
        return self['f_y']
    
    @f_y.setter
    def f_y(self, value:float):
        self['f_y'] = value

    @property
    def f_z(self) -> float:
        return self['f_z']
    
    @f_z.setter
    def f_z(self, value:float):
        self['f_z'] = value

    @property
    def tau_x(self) -> float:
        return self['tau_x']
    
    @tau_x.setter
    def tau_x(self, value:float):
        self['tau_x'] = value

    @property
    def tau_y(self) -> float:
        return self['tau_y']
    
    @tau_y.setter
    def tau_y(self, value:float):
        self['tau_y'] = value

    @property
    def tau_z(self) -> float:
        return self['tau_z']
    
    @tau_z.setter
    def tau_z(self, value:float):
        self['tau_z'] = value

    @property
    def uvn(self) -> tuple[float, float, float]:
        return self.f_x, self.f_y, self.tau_z

def test():
    f1 = GeneralizedForces(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    f2 = GeneralizedForces(1, 2, 3, 4, 5, 6)
    f_add = f1 + f2
    f_sub = f1 - f2
    print("All components should be 0: ", -4 * f1 + 2 * f_add + 4 * f_sub / 2)
    ax = f1.plot(0, 0)
    f2.plot(0, 0, ax=ax)
    plt.show()

if __name__ == "__main__":
    test()

