from nav_env.geometry.stochastic import StochasticVectorFactory
from nav_env.wind.wind_vector import WindVector, DEFAULT_ANGLE_REFERENCE
from nav_env.wind.wind_source import UniformWindSource
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random, math, numpy as np, matplotlib.pyplot as plt

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
    
    def plot_multivariate(self, *args, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        #Parameters to set
        mu_x = self.intensity
        sigma_x = self.sigma_intensity

        mu_y = self.direction
        sigma_y = self.sigma_direction_rad

        #Create grid and multivariate normal
        N = 3
        x = np.linspace(mu_x-N*sigma_x,mu_x+N*sigma_x,100)
        y = np.linspace(mu_y-N*sigma_y,mu_y+N*sigma_y,100)
        X, Y = np.meshgrid(x,y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        rv = multivariate_normal([mu_x, mu_y], [[sigma_x**2, 0], [0, sigma_y**2]])

        #Make a 3D plot
        ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
        ax.set_xlabel('Wind Intensity [m/s]')
        ax.set_ylabel('Wind Direction [rad]')
        ax.set_zlabel('pdf [-]')
        return ax
    
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
        self._nominal_wind_vector = WindVector((0, 0), vector=(self._nominal_velocity_x, self._nominal_velocity_y))

    def __call__(self, nominal=False) -> UniformWindSource:
        """
        Create a stochastic uniform wind source.
        """
        # print(f"{self._nominal_wind_vector.direction:.2f}, {self._nominal_wind_vector.intensity:.2f}")
        
        if nominal:
            return  UniformWindSource(self._nominal_velocity_x, self._nominal_velocity_y, domain=self.domain)

        max_iter, i = 100, 0
        while True:
            i+=1
            stochastic_intensity = random.gauss(self._nominal_wind_vector.intensity, self._sigma_intensity)
            if stochastic_intensity > 0:
                break
            elif i>= max_iter:
                raise ValueError("Intensity is always zero, try to decrease intensity uncertainty.")
        
        stochastic_direction = random.gauss(self._nominal_wind_vector.direction, self._sigma_direction_rad)
        stochastic_wind_vector = WindVector(self._nominal_wind_vector.position, intensity=stochastic_intensity, direction=stochastic_direction)
            
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
    plt.close()

    sigma:dict={'intensity':3, 'angle':15*pi/180}
    wind_factory = StochasticWindVectorFactory(nominal_position=(0, 0), nominal_vector=(10, 5), sigma_intensity=sigma['intensity'], sigma_direction_rad=sigma['angle'])
    ax = wind_factory.plot_multivariate()
    ax.view_init(elev=31, azim=-45, roll=0)
    plt.show()

if __name__ == "__main__":
    # test_stochastic_wind_vector_factory()
    test_stochastic_uniform_wind_source_factory()