from nav_env.geometry.stochastic import StochasticVectorFactory
from nav_env.wind.wind_vector import WindVector
from nav_env.wind.wind_source import UniformWindSource
import random

class StochasticWindVectorFactory(StochasticVectorFactory):
    """
    Factory to create stochastic wind vectors.
    """
    def __init__(self, nominal_position: tuple[float, float], nominal_vector: tuple[float, float], sigma_intensity:float, sigma_direction_rad:float):
        super().__init__(nominal_position, nominal_vector, sigma_intensity, sigma_direction_rad)

    def __call__(self) -> WindVector:
        """
        Create a stochastic wind vector.
        """
        stochastic_vector = super().__call__()
        return WindVector(stochastic_vector.position, vector=stochastic_vector.vector)
    
class StochasticUniformWindSourceFactory(UniformWindSource):
    """
    Used as a factory to generate perturbed uniform wind sources, based on a nominal wind vector and standard deviations on intensity and direction.
    """
    def __init__(self, nominal_velocity_x:float=0., nominal_velocity_y:float=0., sigma_intensity:float=0., sigma_direction_rad:float=0.):
        super().__init__(nominal_velocity_x, nominal_velocity_y)
        self._sigma_intensity = sigma_intensity
        self._sigma_direction_rad = sigma_direction_rad
        self._nominal_velocity_x = nominal_velocity_x
        self._nominal_velocity_y = nominal_velocity_y

    def __call__(self) -> UniformWindSource:
        """
        Create a stochastic uniform wind source.
        """
        wind_vector = WindVector((0, 0), vector=(self._nominal_velocity_x, self._nominal_velocity_y))
        stochastic_intensity = random.gauss(wind_vector.intensity, self._sigma_intensity)
        stochastic_direction = random.gauss(wind_vector.direction, self._sigma_direction_rad)
        stochastic_wind_vector = WindVector(wind_vector.position, intensity=stochastic_intensity, direction=stochastic_direction)
        return UniformWindSource(stochastic_wind_vector.vx, stochastic_wind_vector.vy, domain=self.domain)
    
def test_stochastic_wind_vector_factory():
    import matplotlib.pyplot as plt
    from math import pi
    wind_factory = StochasticWindVectorFactory((2, 3), (-5, -5), 2, 45*pi/180)
    
    N = 100
    ax = None
    for _ in range(N):
        stochastic_wind = wind_factory()
        ax = stochastic_wind.quiver(ax=ax, color='red', angles='xy', scale_units='xy', scale=1)
        print(stochastic_wind)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.show()

def test_stochastic_uniform_wind_source_factory():
    import matplotlib.pyplot as plt
    from math import pi
    uniform_wind_source_factory = StochasticUniformWindSourceFactory(1, -2, 0.5, 45*pi/180)
    lim = ((-10, -10), (10, 10))
    
    N = 10
    ax = None
    for _ in range(N):
        ax = uniform_wind_source_factory().quiver(lim, ax=ax, angles='xy', scale_units='xy', scale=1, nx=5, ny=5)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.show()

if __name__ == "__main__":
    # test_stochastic_wind_vector_factory()
    test_stochastic_uniform_wind_source_factory()