from abc import ABC, abstractmethod
from nav_env.environment.environment import NavigationEnvironment
import numpy as np
import matplotlib.pyplot as plt

class RiskMetric(ABC):
    def __init__(self, env:NavigationEnvironment=None):
        self._env = env or NavigationEnvironment()

    @abstractmethod
    def calculate(self, *args, x:float=None, y:float=None, **kwargs) -> float:
        pass
    
    def plot(self, *args, ax=None, lim=((-100, -100), (100, 100)), res=(20, 20), colorbar:bool=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        x = np.linspace(lim[0][0], lim[1][0], res[0])
        y = np.linspace(lim[0][1], lim[1][1], res[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = self.calculate(*args, x=x[i], y=y[j])

        cont = ax.contourf(X, Y, Z, **kwargs)
        if colorbar:
            plt.colorbar(cont, ax=ax)
            
        ax.set_xlim((lim[0][0], lim[1][0]))
        ax.set_ylim((lim[0][1], lim[1][1]))
        return ax

    def __repr__(self):
        return f"{type(self).__name__} risk metric"
    
    @property
    def env(self):
        return self._env

