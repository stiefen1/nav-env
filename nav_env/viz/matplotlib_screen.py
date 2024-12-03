from nav_env.environment.environment import NavigationEnvironment
import time, matplotlib.pyplot as plt
from nav_env.ships.ship import ShipWithDynamicsBase
from nav_env.obstacles.ship import ShipWithKinematics

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
    def __init__(self, env:NavigationEnvironment, lim:tuple[tuple, tuple]=((-10, -10), (10, 10)), ax=None):
        self._env = env
        self._lim_x = (lim[0][0], lim[1][0])
        self._lim_y = (lim[0][1], lim[1][1])
        self._ax = ax

    def play(self, t0:float=0, tf:float=10, dt:float=0.03, ax=None):
        """
        Play the environment during an interval of time.
        """
        if ax is None:
            _, ax = plt.subplots()

        t_initial = time.time()
        while True:
            ax.cla()
            ax.set_xlim(*self._lim_x)
            ax.set_ylim(*self._lim_y)
            t = time.time() - t_initial
            self._env.step()
            self._env.plot(t+t0, ax=ax)
            ax.set_title(f"t = {t:.2f}")
            plt.pause(dt)

            if t > tf:
                ax.set_title(f"t = {tf:.2f} : Done")
                plt.waitforbuttonpress(120)
                break

            

def test():
    from nav_env.obstacles.obstacles import ObstacleWithKinematics
    from nav_env.obstacles.collection import ObstacleWithKinematicsCollection
    from nav_env.obstacles.ship import ShipWithKinematics
    o1 = ObstacleWithKinematics(lambda t: (t, -t, t*10), xy=[(0, 0), (2, 0), (2, 2), (0, 2)]).rotate(45).translate(0., 9.)
    o2 = ObstacleWithKinematics(lambda t: (t, t, t*20), xy=[(0, 0), (2, 0), (2, 2), (0, 2)]).rotate(45).translate(0., 0.)
    ts1 = ShipWithKinematics(length=20, width=10, ratio=7/9, p0=(-10, 10, 0), v0=(1, -1, 0), make_heading_consistent=True)
    coll = ObstacleWithKinematicsCollection([o1, o2, ts1])

    env = NavigationEnvironment(obstacles=coll)
    screen = MatplotlibScreen(env)
    screen.play()

if __name__ == "__main__":
    test()
