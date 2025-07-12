from nav_env.environment.environment import NavigationEnvironment
from nav_env.risk.monitor import RiskMonitor
from nav_env.simulation.results import SimulationRecord
from alive_progress import alive_bar
import os, matplotlib.pyplot as plt

class Simulator:
    def __init__(self, env:NavigationEnvironment, monitor:RiskMonitor=None):
        self._env = env
        self._record = SimulationRecord(env, monitor=monitor)
        self.times = None

    def run(self,
            tf:float=10,
            dt:float=None,
            record_results_dt:float=None,
            **kwargs
            ):
        self._env.dt = dt
        N = int(tf//self._env.dt)
        self.times = []

        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')

        print("\nSimulation in progress..")
        with alive_bar(N) as bar:
            while self._env.t <= tf:
                if record_results_dt is not None and (self._env.t % record_results_dt < 0.99*self._env.dt):
                    self._record()
                
                self._env.step()
                self.times.append(self._env.t)
                bar()

    def replay(self, x_lim:tuple, y_lim:tuple, *args, t0=-float('inf'), tf=float('inf'), ax=None, speed:float=1.0, **kwargs) -> None:
        assert self.times is not None, f"You must run your simulation once before trying to replay it."
        
        if ax is None:
            _, ax = plt.subplots()

        # Start replay
        # speed_count = 1.0
        for i, t in enumerate(self.times):
            if t < t0 or t > tf:
                continue
            # if speed_count < speed:
            #     speed_count += 1.0
            #     continue
            
            ax.cla()   
            ax = self._env.plot_at_idx(i, x_lim, y_lim, ax=ax)
            ax.set_title(f"t = {t:.2f}")
            plt.pause(self._env.dt/speed)
            # speed_count = 1.0
            
        plt.waitforbuttonpress(timeout=100)
        plt.close()
            

    @property
    def record(self) -> SimulationRecord:
        return self._record

        

def test():
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.ships.ship import Ship, SimpleShip
    from nav_env.ships.states import States3
    from nav_env.simulation.integration import Euler

    from nav_env.risk.ttg import TTG, TTGMaxWorsening
    from nav_env.risk.ddv import DDV2
    from nav_env.obstacles.obstacles import Circle, Obstacle, Ellipse
    from nav_env.risk.monitor import RiskMonitor
    from nav_env.wind.wind_source import UniformWindSource
    import matplotlib.pyplot as plt

    # ship1 = Ship(name='ship1')
    # ship2 = Ship(name='ship2')
    # obs = Circle(50, 30, 30)
    # wind_source = UniformWindSource(30, 50)

    # Shore (Made of obstacles)
    island1 = Obstacle(xy=[(-400, -300), (-200, -300), (-100, -150), (-150, -50), (0, 200), (-400, 300)]).buffer(-100).buffer(50).rotate(-10).translate(0, 50)
    island2 = Obstacle(xy=[(200, -300), (400, -300), (300, -150), (250, -50), (200, 200), (0, 300)]).buffer(-50).buffer(25).rotate(10).translate(0, -150)
    
    # Ships
    os1 = SimpleShip(States3(-250., -500., -20., 7.5, 18., 0.), domain=Ellipse(0, 0, 200, 100), name="OS1")
    os2 = SimpleShip(States3(0., -500., 0., 0, 18., 0.), domain=Ellipse(0, 0, 100, 200), name="OS2")

    # Wind
    wind = UniformWindSource(-10, -30)

    # Environment
    env = Env(
        shore=[island1, island2],
        own_ships=[os1, os2],#, os2],
        wind_source=wind,
        )
    
    # Monitor
    monitor = RiskMonitor([TTG, DDV2])
    sim = Simulator(env=env, monitor=monitor)
    # sim = Simulator(NavigationEnvironment(own_ships=[ship1, ship2], wind_source=wind_source, shore=[obs], dt=0.5), monitor=RiskMonitor([TTG]))
    sim.run(tf=40, record_results_dt=2)

    sim.replay(x_lim=(-1000, 1000), y_lim=(-1000, 1000), t0=10, tf=25, speed=3)

    ax = env.shore.plot()
    os1.plot(ax=ax, params={'enveloppe':1, 'domain':1})
    ax.scatter(sim.record['OS1']['states']['x'], sim.record['OS1']['states']['y'], c=sim.record['OS1']['risks']['TTG'])
    ax.axis('equal')
    plt.show()
    sim.record.save('test.csv')
    new_sim_record = SimulationRecord(path_to_existing_data='test.csv')
    ax = env.shore.plot()
    os1.plot(params={'enveloppe':1, 'domain':1}, ax=ax)
    os2.plot(params={'enveloppe':1, 'domain':1}, ax=ax)

    ax.scatter(new_sim_record['OS1']['states']['x'], new_sim_record['OS1']['states']['y'], c=new_sim_record['OS1']['risks']['DDV2'])
    ax.scatter(new_sim_record['OS2']['states']['x'], new_sim_record['OS2']['states']['y'], c=new_sim_record['OS2']['risks']['DDV2'])
    ax.axis('equal')
    plt.show()


if __name__=="__main__":
    test()
