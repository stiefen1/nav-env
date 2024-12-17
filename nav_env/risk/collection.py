from nav_env.risk.risk import RiskMetric
from typing import Type
import time

class RiskCollection:
    def __init__(self, risks: list[RiskMetric] = None, weights: list[float] = None):
        assert isinstance(risks, list), f"Expected list got {type(risks).__name__}"
        self._risks = risks or []
        self._weights = weights or [1] * len(self._risks)

    def calculate(self, *args, **kwargs) -> float:
        """
        Calculate the risk.
        """
        return sum(self.calculate_separately(*args, **kwargs))
    
    def calculate_separately(self, *args, **kwargs) -> list[float]:
        """ 
        Calculate the risks separately.
        """
        return [w * risk.calculate(*args, **kwargs) for w, risk in zip(self._weights, self._risks)]


