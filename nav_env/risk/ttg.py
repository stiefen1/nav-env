from nav_env.risk.risk import RiskMetric
from nav_env.ships.ship import Ship
from nav_env.ships.states import States3
from nav_env.environment.environment import NavigationEnvironment
from nav_env.wind.wind_source import UniformWindSource
from nav_env.water.water_source import UniformWaterSource
import multiprocessing as mp, numpy as np, matplotlib.pyplot as plt
from nav_env.geometry.line import Line
from shapely import Point
import warnings
from copy import deepcopy


"""
Ideal usage:

risk = TTG(env)
ttg = risk.calculate(ship, t_max=100., precision_sec=1.)
print(f"TTG: {ttg:.2f}")

"""
class TTG(RiskMetric):
    def __init__(self, env:NavigationEnvironment):
        warnings.warn(UserWarning("This class must be fixed, it is not working properly."))
        super().__init__(env)
    
    def calculate(self, ship:Ship, t_max:float=100., precision_sec:float=1., **kwargs) -> float:
        """
        Calculate the Time To Grounding (TTG) for a ship.

        Args:
            ship (Ship): Ship object.
            t_max (float, optional): Maximum time to simulate. Defaults to 100..
            precision_sec (float, optional): Period at which we check for collision with the environment. Defaults to 1..
        """
        ship_copy = deepcopy(ship)
        dt = ship_copy.integrator.dt
        t:float = 0.
        # average_step_duration = 0.
        # average_loop_duration = 0.
        n = 0
        while t < t_max:
            # start_loop = time.time()
            if t % precision_sec < dt:
                ship_copy.update_enveloppe_from_accumulation() # WE ONLY UPDATE THE ENVELOPPE BEFORE CHECKING FOR COLLISIONS
                if ship_copy.collide(self.env.shore):
                    return t
            
            # start = time.time()

            # TODO: Optimize computational time, typically step() takes on average 0.2ms, and overall loop takes 0.25ms -> C++, Cython ?
            
            # Setting update_enveloppe to False to avoid updating the enveloppe at each time step. We only want to update it when checking for collisions.
            ship_copy.step(self.env.wind_source(ship_copy.states.xy), self.env.water_source(ship_copy.states.xy), update_enveloppe=False) 
            # end = time.time()
            # average_step_duration += end - start
            # average_loop_duration += time.time() - start_loop
            # n += 1
            t += dt

        # print(f"Average step duration: {1000*average_step_duration / n:.2f}ms | Average loop duration: {1000*average_loop_duration / n:.2f}ms")
        return t_max
    
    def plot(self, ax=None, **kwargs):
        pass
    
    
# TODO: Sometimes TTG are 0 when running 100 environments in parallel, why ?
class ParallelTTG:
    def __init__(self, num_workers: int = 4):
        warnings.warn(UserWarning("This class must be fixed, it is not working properly."))
        self.num_workers = num_workers

    def run_parallel(self, ship: Ship, envs: NavigationEnvironment, t_max: float = 100., precision_sec: float = 1., **kwargs) -> list[float]:
        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.starmap(self._run_ttg, [(ship, env, t_max, precision_sec, kwargs) for env in envs])
        return results

    def _run_ttg(self, ship: Ship, env: NavigationEnvironment, t_max: float, precision_sec: float, kwargs) -> float:
        ttg = TTG(env)
        return ttg.calculate(ship, t_max, precision_sec, **kwargs)
    

def test_parallel_ttg():
    import time
    from nav_env.simulation.integration import Euler
    from nav_env.wind.stochastic import StochasticUniformWindSourceFactory
    from math import pi
    from nav_env.obstacles.obstacles import Circle
    from nav_env.obstacles.collection import ObstacleCollection
    import matplotlib.pyplot as plt

    shore = ObstacleCollection([Circle(200., -200., 50.)])
    wind_source_factory = StochasticUniformWindSourceFactory(50, -50, 10, 45*pi/180)
    ship = Ship(integrator=Euler(1.))
    envs = [NavigationEnvironment(wind_source=wind_source_factory(), shore=shore) for _ in range(10000)]  # Create a list of ships
    envs[0].plot(0, lim=((-300, -300), (300, 300)))
    plt.show()
    parallel_ttg = ParallelTTG(num_workers=10)
    start = time.time()
    results = parallel_ttg.run_parallel(ship, envs, t_max=50., precision_sec=5.)
    print(f"TTG: {results} computed in {time.time() - start:.2f}s | min: {min(results):.2f}, max: {max(results):.2f}")




############################################ In this section we implement TTG for static environments ############################################

# Wind must be uniform and constant in time
# Water must be uniform and constant in time
# Heading must always be the same

class TTGUnderConstantUniformPerturbations(RiskMetric):
    def __init__(self, ship:Ship, env:NavigationEnvironment, t_max: float = 100., precision_sec: float = 1., **kwargs):
        assert t_max > 0, "t_max must be greater than 0"
        assert precision_sec > 0, "precision_sec must be greater than 0"
        
        if not isinstance(env.wind_source, UniformWindSource):
            UserWarning(f"Wind source must be of type UniformWindSource, got {type(env.wind_source).__name__}")
        if not isinstance(env.water_source, UniformWaterSource):
            UserWarning(f"Water source must be of type UniformWaterSource, got {type(env.water_source).__name__}")
        super().__init__(env)

        self._ship = ship
        self._t_max = t_max
        self._precision_sec = precision_sec
        self._compute_trajectory()

    def _compute_trajectory(self):
        dt = self._ship.integrator.dt
        t:float = 0.
        self._ship.states.x, self._ship.states.y = 0., 0.
        traj = [(*self._ship.states.xy, t)]
        while t < self._t_max:
            # Setting update_enveloppe to False to avoid updating the enveloppe at each time step. We only want to update it when checking for collisions.
            self._ship.step(self.env.wind_source(self._ship.states.xy), self.env.water_source(self._ship.states.xy), update_enveloppe=False)
            t += dt
            if t % self._precision_sec < dt:
                traj.append((*self._ship.states.xy, t))

        self._line = Line(traj)

    def calculate(self, ship_x, ship_y) -> float:
        translated_shore = self.env.shore.translate(-ship_x, -ship_y)
        
        # Check if the ship is already in collision with the shore
        for obs in translated_shore:
            if obs._geometry.contains(Point(0, 0)):
                return 0.
            
        # Check for intersection between the ship's trajectory and the shore
        intersections = [self._line._geometry.intersection(obs._geometry.exterior) for obs in translated_shore]
        times = []
        for intersection in intersections:
            if intersection.is_empty:
                continue
            try:
                # Intersection is a multipoint
                for geom in intersection.geoms:
                    times.append(geom.z)
            except:
                # Intersection is a point
                times.append(intersection.z)
        return min(times) if times else self._t_max


    def plot(self, lim=((-100, -100), (100, 100)), res=(100, 100), ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        x = np.linspace(lim[0][0], lim[1][0], res[0])
        y = np.linspace(lim[0][1], lim[1][1], res[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = self.calculate(x[i], y[j])

        cont = ax.contourf(X, Y, Z, **kwargs)
        self._line.plot('--r', ax=ax)
        self._ship.reset()
        self._ship.plot(ax=ax, keys=['enveloppe', 'velocity'])
        self._env.shore.plot('r', ax=ax)
        plt.colorbar(cont, ax=ax)
        ax.set_xlim((lim[0][0], lim[1][0]))
        ax.set_ylim((lim[0][1], lim[1][1]))
        return ax
    
    def plot3(self, lim=((-100, -100), (100, 100)), res=(100, 100), ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

        x = np.linspace(lim[0][0], lim[1][0], res[0])
        y = np.linspace(lim[0][1], lim[1][1], res[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j, i] = self.calculate(x[i], y[j])

        ax.plot_surface(X, Y, Z, **kwargs)
        return ax

    @property
    def line(self) -> Line:
        return self._line


def test_ttg_under_constant_uniform_perturbations():
    import time
    from nav_env.simulation.integration import Euler
    from nav_env.wind.wind_source import UniformWindSource
    from nav_env.obstacles.obstacles import Circle, Ellipse
    from nav_env.obstacles.collection import ObstacleCollection
    import matplotlib.pyplot as plt

    shore = ObstacleCollection([Circle(300., -0., 50.), Ellipse(800, -200, 40, 80)])
    ship = Ship(integrator=Euler(1.))
    wind_source = UniformWindSource(50, -50)
    env = NavigationEnvironment(wind_source=wind_source, shore=shore)  # Create a list of ships
    start = time.time()
    ttg_uniform = TTGUnderConstantUniformPerturbations(ship, env, t_max=100., precision_sec=1.)
    print(f"Computed in {1000*(time.time() - start):.2f}ms")
    ax = env.plot(0, lim=((-300, -300), (300, 300)))
    ttg_uniform.line.plot(ax=ax)
    plt.show()
    
    # print(ttg_uniform.line)
    start = time.time()
    # print([ttg_uniform.line.intersection(obs.exterior) for obs in shore])
    print(ttg_uniform.calculate(0, 0))
    print(f"Computed in {1000*(time.time() - start):.2f}ms")

def show_ttg_contour_under_constant_uniform_perturbations():
    import numpy as np
    from matplotlib import pyplot as plt
    from nav_env.obstacles.collection import ObstacleCollection
    from nav_env.obstacles.obstacles import Circle, Ellipse
    from nav_env.ships.ship import Ship
    from nav_env.ships.states import States3	
    from nav_env.simulation.integration import Euler
    from nav_env.environment.environment import NavigationEnvironment
    from nav_env.wind.wind_source import UniformWindSource
    from nav_env.risk.ttg import TTGUnderConstantUniformPerturbations

    shore = ObstacleCollection([Circle(300., -0., 50.), Ellipse(400, -200, 40, 80)])
    ship = Ship(integrator=Euler(1.), states=States3(0., 0., 60., 30., 10., 0.))
    wind_source = UniformWindSource(50, -50)
    env = NavigationEnvironment(wind_source=wind_source, shore=shore)  # Create a list of ships
    ttg_uniform = TTGUnderConstantUniformPerturbations(ship, env, t_max=100., precision_sec=0.1)
    lim = ((-200, -800), (1200, 600))
    fig, ax = plt.subplots(1, 2)
    ttg_uniform.plot(lim=lim, ax=ax[0], cmap='YlGnBu', levels=50)
    env.plot(0, lim=lim, ax=ax[1])
    ax[1].set_aspect('equal')
    ax[0].set_aspect('equal')
    plt.show()

    # 3d Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ttg_uniform.plot3(lim=lim, ax=ax, cmap='YlGnBu')
    plt.show()






    


# Example usage
if __name__ == "__main__":
    # test_parallel_ttg()
    # test_ttg_under_constant_uniform_perturbations()
    show_ttg_contour_under_constant_uniform_perturbations()