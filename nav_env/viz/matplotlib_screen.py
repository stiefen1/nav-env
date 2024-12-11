from nav_env.environment.environment import NavigationEnvironment
import matplotlib.pyplot as plt, time
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
             own_ships_verbose=['enveloppe', 'frame', 'acceleration', 'velocity', 'forces'],
             target_ships_verbose=['enveloppe'],
             **kwargs
             ):
        """
        Play the environment during an interval of time.
        """
        self._env.dt = dt # Enforce the time step for the whole environment

        if self._monitor is not None:
            self.play_with_monitor(tf, ax, own_ships_verbose, target_ships_verbose, **kwargs)
        else:
            self.play_without_monitor(tf, ax, own_ships_verbose, target_ships_verbose, **kwargs)

    def play_with_monitor(self,
                          tf:float=10,
                          ax=None,
                          own_ships_verbose=['enveloppe', 'frame', 'acceleration', 'velocity', 'forces'],
                          target_ships_verbose=['enveloppe'],
                          **kwargs):
        """
        Play the environment during an interval of time.

        WARNING: This function requires the whole environment to be pickable to run the monitor in a separate process.
        """
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[1].grid()
        # ax[1].legend('TTG', 'Distance')

        manager = mp.Manager()
        shared_env_dict = manager.dict(self._env.to_dict()) 
        result_queue = mp.Queue()
        risk_process = mp.Process(target=self._monitor.monitor, args=(shared_env_dict, result_queue))
        risk_process.start()

        while True:
            loop_start = time.time()
            ax[0].cla()
            ax[0].set_xlim(*self._lim_x)
            ax[0].set_ylim(*self._lim_y)
            self._env.step()
            self._env.plot(self._lim, own_ship_physics=own_ships_verbose, target_ship_physics=target_ships_verbose, ax=ax[0])
            ax[0].set_title(f"t = {self._env.t:.2f}")

            shared_env_dict.update(self._env.to_dict())

            if not result_queue.empty():
                risk_values = result_queue.get()
                ax[1].plot(risk_values[0], risk_values[1], 'ro')
                ax[1].plot(risk_values[0], risk_values[2], 'bo')
                ax[1].legend(self._monitor.legend())

            if self._env.t > tf:
                ax[0].set_title(f"t = {tf:.2f} : Done")
                risk_process.terminate()
                print("Simulation done. Press any button to exit.")
                plt.waitforbuttonpress(120)
                break

            loop_end = time.time()
            plt.pause(max(1e-9, self._env.dt - (loop_end - loop_start)))

    def play_without_monitor(self,
                          tf:float=10,
                          ax=None,
                          own_ships_verbose=['enveloppe', 'frame', 'acceleration', 'velocity', 'forces'],
                          target_ships_verbose=['enveloppe'],
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
            self._env.plot(self._lim, own_ship_physics=own_ships_verbose, target_ship_physics=target_ships_verbose, ax=ax)
            ax.set_title(f"t = {self._env.t:.2f}")

            if self._env.t > tf:
                ax.set_title(f"t = {tf:.2f} : Done")
                print("Simulation done. Press any button to exit.")
                plt.waitforbuttonpress(120)
                break

            loop_end = time.time()
            plt.pause(max(1e-9, self._env.dt - (loop_end - loop_start)))

    @property
    def lim_x(self) -> tuple:
        return self._lim_x
    
    @property
    def lim_y(self) -> tuple:
        return self._lim_y  


 

def test():
    from nav_env.obstacles.obstacles import ObstacleWithKinematics
    from nav_env.ships.sailing_ship import SailingShip
    from nav_env.ships.states import States3
    from nav_env.viz.matplotlib_screen import MatplotlibScreen
    from nav_env.environment.environment import NavigationEnvironment

    def o1_pose(t:float) -> States3:
        return States3(-t, t, t*10)

    def o2_pose(t:float) -> States3:
        return States3(t, -t, t*20)

    o1 = ObstacleWithKinematics(o1_pose, xy=[(0, 0), (2, 0), (2, 2), (0, 2)])
    o2 = ObstacleWithKinematics(o2_pose, xy=[(0, 0), (2, 0), (2, 2), (0, 2)])
    ts1 = SailingShip(length=20, width=10, ratio=7/9, initial_state=States3(-10, 10, 0, 1, -1, 0))
    coll = [o1, o2, ts1]

    env = NavigationEnvironment(obstacles=coll)
    screen = MatplotlibScreen(env)
    screen.play()

if __name__ == "__main__":
    test()
