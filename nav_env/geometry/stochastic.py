from nav_env.geometry.vector import Vector
from nav_env.geometry.vector_source import VectorSource
import random
from typing import Callable
from math import pi

"""
Stationarity means:
- mean is constant
- variance is constant
- There is no seasonality -> The signal is not periodic
"""

class StochasticVectorFactory(Vector):
    def __init__(self, nominal_position: tuple[float, float], nominal_vector: tuple[float, float], sigma_intensity:float, sigma_direction_rad:float):
        super().__init__(nominal_position, vector=nominal_vector)
        self._sigma_intensity = sigma_intensity
        self._sigma_direction_rad = sigma_direction_rad

    def __call__(self, distribution:str='gauss') -> Vector:
        if distribution == 'gauss':
            distribution = random.gauss
        elif distribution == 'uniform':
            distribution = random.uniform
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        stochastic_intensity = distribution(self.intensity, self._sigma_intensity)
        stochastic_direction_rad = distribution(self.direction, self._sigma_direction_rad)
        assert stochastic_intensity >= 0, f"Intensity is negative: {stochastic_intensity}"
        return Vector(self.position, intensity=stochastic_intensity, direction=stochastic_direction_rad)
    
    def __repr__(self):
        return super().__repr__() + f" sigma_intensity={self._sigma_intensity:.3f}, sigma_direction_rad={self._sigma_direction_rad:.3f}"
    
    @property
    def sigma_intensity(self) -> float:
        return self._sigma_intensity
    
    @property
    def sigma_direction_rad(self) -> float:
        return self._sigma_direction_rad
    
    @property
    def sigma_direction_deg(self) -> float:
        return self._sigma_direction_rad * 180 / pi
    
    
def test_stochastic_vector_factory():
    import matplotlib.pyplot as plt
    factory = StochasticVectorFactory((2, 3), (-5, -5), 2, 45*pi/180)
    
    N = 100
    ax = None
    for _ in range(N):
        stochastic_vector = factory()
        ax = stochastic_vector.quiver(ax=ax, color='red', angles='xy', scale_units='xy', scale=1)
        print(stochastic_vector)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.show()




def test_stochastic_vector_source_factory():
    pass

if __name__ == "__main__":
    test_stochastic_vector_factory()
    

    