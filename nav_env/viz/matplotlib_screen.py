from nav_env.environment.environment import NavigationEnvironment
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec, time
import multiprocessing as mp
from nav_env.risk.monitor import RiskMonitor
from nav_env.ships.states import States3



# Maybe not the best architecture
# Ideally we would have one class that runs simulation
# and another class that plots the simulation (taking the simulation as input)
# Dans stable-baselines3, we consider the physical envelope of our agent as part
# of the environment, and the agent is only the policy.
# Simulation is done via:¨

"""
env = Env()
policy = Policy()
obs = env.reset()
T = 100
for t in range(T):
    action, states = policy.get_action(obs)
    obs, reward, dones, info = env.step(action)
    env.render()

"""

# TODO: Make Screen able to play any sort of playable object, not only environments. 
# This would allow us to play a simulation of a single ship.
# We could add element to the screen, in a "PlayableCollection" object for instance.

class MatplotlibScreen:
    def __init__(self, env:NavigationEnvironment, monitor:RiskMonitor=None, lim:tuple[tuple, tuple]=((-10, -10), (10, 10)), scale:float=1, ax=None):
        self._env = env
        self._monitor = monitor
        self._lim = lim
        self._lim_x = (lim[0][0], lim[1][0])
        self._lim_y = (lim[0][1], lim[1][1])
        self._dx = abs(self._lim_x[1] - self._lim_x[0]) # Maybe will be used for scaling vectors
        self._dy = abs(self._lim_y[1] - self._lim_y[0])
        self._ax = ax
        self._scale = scale

    def play(self,
             tf:float=10,
             dt:float=None,
             ax=None,
             own_ships_verbose={'enveloppe':1, 'frame':1, 'acceleration':1, 'velocity':1, 'forces':1},
             target_ships_verbose={'enveloppe':1},
             speed:float=1., 
             **kwargs
             ):
        """
        Play the environment during an interval of time.
        """
        self._env.dt = dt # Enforce the time step for the whole environment

        if self._monitor is not None:
            self.play_with_monitor(tf, ax, own_ships_verbose, target_ships_verbose, speed=speed, **kwargs)
        else:
            self.play_without_monitor(tf, ax, own_ships_verbose, target_ships_verbose, speed=speed, **kwargs)

    def play_with_monitor(self,
                          tf:float=10,
                          ax=None,
                          own_ships_verbose={'enveloppe':1, 'frame':1, 'acceleration':1, 'velocity':1, 'forces':1},
                          target_ships_verbose={'enveloppe':1},
                          buffer:int=10,
                          speed:float=1.0,
                          **kwargs):
        """
        Play the environment during an interval of time.

        WARNING: This function requires the whole environment to be pickable to run the monitor in a separate process.
        """
        N_ship = len(self._env.own_ships)

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[1].cla()
        ax[1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        gs = gridspec.GridSpecFromSubplotSpec(N_ship, 1, subplot_spec=ax[1].get_subplotspec())
        sub_axes = [fig.add_subplot(gs[i]) for i in range(N_ship)]
        for i, ship in enumerate(self._env.own_ships):
            sub_axes[i].grid()
            sub_axes[i].set_title(ship.name)
            sub_axes[i].tick_params(bottom=False, labelbottom=False)
            sub_axes[i].legend(self._monitor.legend())
        sub_axes[-1].tick_params(bottom=True, labelbottom=True)
        legend = self._monitor.legend()

        manager = mp.Manager()
        shared_env_dict = manager.dict(self._env.to_dict()) 
        result_queue = mp.Queue()
        risk_process = mp.Process(target=self._monitor.monitor, args=(shared_env_dict, result_queue))
        risk_process.start()

        list_of_risks_for_ships = []
        for i in range(N_ship):
            list_of_risks_for_ship_i = []
            for j in range(len(self._monitor)):
                list_of_risk_j_for_ship_i = []
                list_of_risks_for_ship_i.append(list_of_risk_j_for_ship_i)
            list_of_risks_for_ships.append(list_of_risks_for_ship_i)

        times = []

        while True:
            loop_start = time.time()
            ax[0].cla()
            ax[0].set_xlim(*self._lim_x)
            ax[0].set_ylim(*self._lim_y)
            self._env.step()
            self._env.plot(self._lim, own_ships_physics=own_ships_verbose, target_ships_physics=target_ships_verbose, ax=ax[0])
            # risk = self._monitor._risk_classes[0](self._env)
            # risk.plot(self._env.own_ships[0], ax=ax[0], lim=self._lim, res=(20, 20), alpha=0.5)
            ax[0].set_title(f"t = {self._env.t:.2f}")

            shared_env_dict.update(self._env.to_dict())

            if not result_queue.empty():
                risk_values:list = result_queue.get()
                times.append(risk_values.pop(0))
                if len(times) > buffer:
                    times.pop(0)
                for i, risk_for_ship_i in enumerate(risk_values):
                    sub_axes[i].cla()
                    sub_axes[i].set_title(self._env.own_ships[i].name)
                    sub_axes[i].grid()
                    for j, value in enumerate(risk_for_ship_i):
                        list_of_risks_for_ships[i][j].append(value)
                        if len(list_of_risks_for_ships[i][j]) > buffer:
                            list_of_risks_for_ships[i][j].pop(0)
                        sub_axes[i].plot(times, list_of_risks_for_ships[i][j], 'o-', color=f'C{j}')
                    sub_axes[i].legend(legend)
                    sub_axes[i].set_ylim(0, 100)

                    # sub_axes[i].set_ylim([0, 100])

                    
            if self._env.t > tf:
                ax[0].set_title(f"t = {tf:.2f} : Done")
                risk_process.terminate()
                print("Simulation done. Press any button to exit.")
                plt.waitforbuttonpress(120)
                break

            loop_end = time.time()
            plt.pause(max(1e-9, (self._env.dt/speed) - (loop_end - loop_start)))

    def play_without_monitor(self,
                          tf:float=10,
                          ax=None,
                          own_ships_verbose={'enveloppe':1, 'frame':1, 'acceleration':1, 'velocity':1, 'forces':1},
                          target_ships_verbose={'enveloppe':1},
                          speed:float=1.0,
                          **kwargs):
        """
        Play the environment during an interval of time.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.grid()

        while True:
            loop_start = time.time()
            ax.cla()
            ax.set_xlim(*self._lim_x)
            ax.set_ylim(*self._lim_y)
            self._env.step()
            self._env.plot(self._lim, own_ships_physics=own_ships_verbose, target_ships_physics=target_ships_verbose, ax=ax)
            ax.set_title(f"t = {self._env.t:.2f}")

            if self._env.t > tf:
                ax.set_title(f"t = {tf:.2f} : Done")
                print("Simulation done. Press any button to exit.")
                plt.waitforbuttonpress(120)
                break


            loop_end = time.time()
            plt.pause(max(1e-12, (self._env.dt/speed) - (loop_end - loop_start) - 5e-3))

    @property
    def lim_x(self) -> tuple:
        return self._lim_x
    
    @property
    def lim_y(self) -> tuple:
        return self._lim_y  


 

def test():
    from nav_env.obstacles.obstacles import MovingObstacle
    from nav_env.ships.sailing_ship import SailingShip
    from nav_env.ships.states import States3
    from nav_env.viz.matplotlib_screen import MatplotlibScreen
    from nav_env.environment.environment import NavigationEnvironment

    def o1_pose(t:float) -> States3:
        return States3(-t, t, t*10)

    def o2_pose(t:float) -> States3:
        return States3(t, -t, t*20)

    o1 = MovingObstacle(o1_pose, xy=[(0, 0), (2, 0), (2, 2), (0, 2)])
    o2 = MovingObstacle(o2_pose, xy=[(0, 0), (2, 0), (2, 2), (0, 2)])
    ts1 = SailingShip(length=20, width=10, ratio=7/9, initial_state=States3(-10, 10, 0, 1, -1, 0))
    coll = [o1, o2, ts1]

    env = NavigationEnvironment(obstacles=coll)
    screen = MatplotlibScreen(env)
    screen.play()

if __name__ == "__main__":
    test()
