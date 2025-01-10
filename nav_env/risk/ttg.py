from nav_env.risk.risk import RiskMetric
from nav_env.ships.ship import Ship
from nav_env.ships.states import States3
from nav_env.ships.ship import ShipWithDynamicsBase, MovingShip
from nav_env.ships.sailing_ship import SailingShip
from nav_env.simulation.integration import Euler
from nav_env.environment.environment import NavigationEnvironment
from nav_env.wind.wind_source import UniformWindSource
from nav_env.wind.stochastic import StochasticUniformWindSourceFactory
from nav_env.water.water_source import UniformWaterSource
import multiprocessing as mp, numpy as np, matplotlib.pyplot as plt
from nav_env.geometry.line import Line
from shapely import Point
import warnings
from copy import deepcopy
from math import pi
from scipy.optimize import curve_fit


"""
Ideal usage:

risk = TTG(env)
ttg = risk.calculate(ship, t_max=100., precision_sec=1.)
print(f"TTG: {ttg:.2f}")

"""
class TTG(RiskMetric):
    def __init__(self, env:NavigationEnvironment):
        # warnings.warn(UserWarning("This class must be fixed, it is not working properly."))
        super().__init__(env)
    
    def calculate(self, ship:MovingShip, t_max:float=100., precision_sec:float=1., output_final_state:bool=False, **kwargs) -> float | tuple[float, States3]:
        """
        Calculate the Time To Grounding (TTG) for a ship.

        Args:
            ship (Ship): Ship object.
            t_max (float, optional): Maximum time to simulate. Defaults to 100..
            precision_sec (float, optional): Period at which we check for collision with the environment. Defaults to 1..
        """
        if isinstance(ship, ShipWithDynamicsBase):
            # Meaning user has specified an integrator
            ship_copy = Ship(integrator=ship.integrator, states=ship.states)
        elif isinstance(ship, MovingShip):
            # Meaning we only have dt -> default integrator to Euler
            ship_copy = Ship(integrator=Euler(ship.dt), states=ship.states)

            if isinstance(ship, SailingShip):
                # Means the ship is following a function, not necessarily considering speed
                # -> Numerical differentiation to find current speed
                t_curr = ship._t
                state_t_plus_dt = ship.pose_fn(t_curr+ship.dt)
                state_t_minus_dt = ship.pose_fn(t_curr-ship.dt)

                # Compute numerical differentation to approximate speed
                dxdt = (state_t_plus_dt.x - state_t_minus_dt.x)/(2*ship.dt)
                dydt = (state_t_plus_dt.y - state_t_minus_dt.y)/(2*ship.dt)
                dpsidt = (state_t_plus_dt.psi_deg - state_t_minus_dt.psi_deg)/(2*ship.dt)
                
                # Set ship speed
                ship_copy.states.x_dot = dxdt
                ship_copy.states.y_dot = dydt
                ship_copy.states.psi_dot_deg = dpsidt
                
        dt = ship_copy.integrator.dt
        t:float = 0.

        while t < t_max:
            # start_loop = time.time()
            if t % precision_sec < dt:
                ship_copy.update_enveloppe_from_accumulation() # WE ONLY UPDATE THE ENVELOPPE BEFORE CHECKING FOR COLLISIONS
                if ship_copy.collide(self.env.shore):
                    return t

            # TODO: Optimize computational time, typically step() takes on average 0.2ms, and overall loop takes 0.25ms -> C++, Cython ?
            
            # Setting update_enveloppe to False to avoid updating the enveloppe at each time step. We only want to update it when checking for collisions.
            ship_copy.step(self.env.wind_source(ship_copy.states.xy), self.env.water_source(ship_copy.states.xy), update_enveloppe=False) 
            t += dt

        if output_final_state:
            return t_max, ship_copy.states
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
    
class TTGStochasticWind(RiskMetric):
    def __init__(self, env:NavigationEnvironment, num_workers: int = 4):
        warnings.warn(UserWarning("This class must be fixed, it is not working properly."))
        self.num_workers = num_workers
        super().__init__(env)

    def get_results(self, ship:Ship, n:int = 20, t_max: float = 100., precision_sec:float = 1., sigma:dict={'intensity':5, 'angle':30*pi/180}, **kwargs) -> list[float]:
        vec = self.env.wind_source(ship.states.xy)
        wind_source_factory = StochasticUniformWindSourceFactory(vec.vx, vec.vy, sigma['intensity'], sigma['angle'])
        args = [(ship, deepcopy(self.env), t_max, precision_sec, kwargs)] # Nominal case
        winds = [vec]
        for _ in range(n):
            env = deepcopy(self.env)
            env.wind_source = wind_source_factory()
            winds.append(env.wind_source((0., 0.)))
            args.append((ship, env, t_max, precision_sec, kwargs))
        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.starmap(self._run_ttg, args)
        return results, winds
    
    def calculate(self) -> float:
        pass

    def _run_ttg(self, ship: Ship, env: NavigationEnvironment, t_max: float, precision_sec: float, kwargs) -> float:
        ttg = TTG(env)
        return ttg.calculate(ship, t_max, precision_sec, **kwargs)


class TTGMaxWorsening(TTGStochasticWind):
    def __init__(self, env:NavigationEnvironment, num_workers: int = 4):
        warnings.warn(UserWarning("This class must be fixed, it is not working properly."))
        self.num_workers = num_workers
        super().__init__(env, num_workers=num_workers)


    def calculate(self, ship:Ship, n:int = 20, t_max: float = 100., precision_sec:float = 1., sigma:dict={'intensity':5, 'angle':30*pi/180}, **kwargs) -> float:
        results, winds = self.get_results(ship, n=n, t_max=t_max, precision_sec=precision_sec, sigma=sigma, **kwargs)

        # Extract nominal TTG
        nominal = results.pop(0)

        # Worst case
        if nominal == 0:
            max_worsening_wrt_nominal = 0.
        else:
            max_worsening_wrt_nominal = max([nominal - result for result in results]) # / nominal * 100
            # max_worsening_wrt_nominal = min([result for result in results]) # / nominal * 100

        return max_worsening_wrt_nominal

    def _run_ttg(self, ship: Ship, env: NavigationEnvironment, t_max: float, precision_sec: float, kwargs) -> float:
        ttg = TTG(env)
        return ttg.calculate(ship, t_max, precision_sec, **kwargs)
    
    def plot(self, ax=None, **kwargs):
        pass

class TTGWorstCase(TTGStochasticWind):
    def __init__(self, env:NavigationEnvironment, num_workers: int = 4):
        warnings.warn(UserWarning("This class must be fixed, it is not working properly."))
        self.num_workers = num_workers
        super().__init__(env)

    def calculate(self, ship:Ship, n:int = 20, t_max: float = 100., precision_sec:float = 1., sigma:dict={'intensity':5, 'angle':30*pi/180}, **kwargs) -> float:
        results, winds = self.get_results(ship, n=n, t_max=t_max, precision_sec=precision_sec, sigma=sigma, **kwargs)

        # Worst case
        ttg_worst_case = min(results)

        return ttg_worst_case

    def plot(self, ax=None, **kwargs):
        pass


class TTGExpectation(TTGStochasticWind):
    def __init__(self, env:NavigationEnvironment, num_workers: int = 4):
        warnings.warn(UserWarning("This class must be fixed, it is not working properly."))
        self.num_workers = num_workers
        super().__init__(env)

    def calculate(self, ship:Ship, n:int = 20, t_max: float = 100., precision_sec:float = 1., sigma:dict={'intensity':5, 'angle':30*pi/180}, **kwargs) -> float:
        results, winds = self.get_results(ship, n=n, t_max=t_max, precision_sec=precision_sec, sigma=sigma, **kwargs)

        # Expectation
        ttg_expected = sum(results)/len(results)
        
        return ttg_expected

    def plot(self, ax=None, **kwargs):
        pass

def quadratic_surface(xy, a, b, c, d, e, f) -> float:
    x, y = xy
    return a*x**2 + b*y**2 + c * x * y  + d * x + e * y +  f

class TTGCurvature(TTGStochasticWind):
    def __init__(self, env:NavigationEnvironment, num_workers: int = 4):
        warnings.warn(UserWarning("This class must be fixed, it is not working properly."))
        self.num_workers = num_workers
        super().__init__(env)

    def calculate(self, ship:Ship, n:int = 20, t_max: float = 100., precision_sec:float = 1., sigma:dict={'intensity':5, 'angle':30*pi/180}, **kwargs) -> float:
        results, winds = self.get_results(ship, n=n, t_max=t_max, precision_sec=precision_sec, sigma=sigma, **kwargs)

        xdata = np.array([(wind.direction, wind.intensity) for wind in winds]).T
        print(len(xdata), len(results))
        popt, pcov = curve_fit(quadratic_surface, xdata, results)
        a, b, c, d, e, f = popt
        curvature_det = np.linalg.det(np.array([[2*a, 2*c], [2*c, 2*b]]))
        print(curvature_det)
        return -curvature_det

    def plot(self, ax=None, **kwargs):
        pass
    

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
    from nav_env.obstacles.obstacles import Circle, Ellipse
    from nav_env.ships.ship import Ship
    from nav_env.ships.states import States3	
    from nav_env.simulation.integration import Euler
    from nav_env.environment.environment import NavigationEnvironment
    from nav_env.wind.wind_source import UniformWindSource
    from nav_env.risk.ttg import TTGUnderConstantUniformPerturbations

    shore = [Circle(300., -0., 50.), Ellipse(400, -200, 40, 80)]
    ship = Ship(integrator=Euler(1.), states=States3(0., 0., 60., 30., 10., 0.))
    wind_source = UniformWindSource(50, -50)
    env = NavigationEnvironment(wind_source=wind_source, shore=shore)  # Create a list of ships
    ttg_uniform = TTGUnderConstantUniformPerturbations(ship, env, t_max=100., precision_sec=0.1)
    lim = ((-200, -800), (1200, 600))
    fig, ax = plt.subplots(1, 2)
    ttg_uniform.plot(lim=lim, ax=ax[0], cmap='YlGnBu', levels=50)
    env.plot(lim=lim, ax=ax[1])
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