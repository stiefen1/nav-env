from abc import ABC, abstractmethod
from nav_env.environment.environment import NavigationEnvironment

class RiskMetric(ABC):
    def __init__(self, env:NavigationEnvironment):
        self._env = env

    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def plot(self, ax=None, **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        return "RiskMetric"
    
    @property
    def env(self):
        return self._env