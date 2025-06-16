from abc import ABC, abstractmethod
from scipy.stats import norm, uniform
from typing import Any

DISTRIBUTION_HASHMAP = {
    "normal": norm,
    "uniform": uniform
}

"""
params = [
    {
        "distribution": "normal",
        "hyper": {
                "loc": 
            }
    }
]
"""

DEFAULT_PARAM = {
    "distribution": "normal",
    "hyper": {
        "loc": 0,   # Mean
        "scale": 0  # Standard deviation
    }
}

class Noise(ABC):
    def __init__(self, params:list[dict]) -> None:    
        """
        params is a list, where each component matches one dimension of the measurement. Noise for each dimension is parametrized
        by a dictionnary which must have the form:
        {
            "distribution": "normal" or "uniform",
            "hyper": {
                "loc": 0,   # Mean
                "scale": 0  # Standard deviation
        }
        """   
        distributions = []
        for param in params:
            distributions.append(DISTRIBUTION_HASHMAP[param["distribution"]](**param["hyper"]))
        self._distributions = tuple(distributions)

    def sample(self) -> tuple:
        noise_values = []
        for distribution in self._distributions:
            noise_values.append(float(distribution.rvs()))
        return tuple(noise_values)
    
    def __call__(self, measurement:Any) -> Any:
        samples = self.sample()
        for sample, key in zip(samples, measurement.__dict__.keys()):
            measurement.__dict__[key] += sample
        return measurement

    @property
    def n(self) -> int:
        return len(self._distributions)
    
    @staticmethod
    def default_noise(n:int) -> "Noise":
        return Noise(n * [DEFAULT_PARAM])
    
def test() -> None:
    
    n = Noise(3 * [DEFAULT_PARAM])
    print(n.sample())

    
if __name__ == "__main__": test()