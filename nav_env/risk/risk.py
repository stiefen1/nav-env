from abc import ABC, abstractmethod
from nav_env.environment.environment import NavigationEnvironment

class RiskMetric(ABC):
    def __init__(self, env:NavigationEnvironment=None):
        self._env = env or NavigationEnvironment()

    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        pass
    
    @abstractmethod
    def plot(self, ax=None, **kwargs):
        pass

    def __repr__(self):
        return f"{type(self).__name__} risk metric"
    
    @property
    def env(self):
        return self._env

