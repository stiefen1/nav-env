from nav_env.environment.environment import NavigationEnvironment
import matplotlib.pyplot as plt, time

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
    def __init__(self, env:NavigationEnvironment, lim:tuple[tuple, tuple]=((-10, -10), (10, 10)), scale:float=1, ax=None):
        self._env = env
        self._lim = lim
        self._lim_x = (lim[0][0], lim[1][0])
        self._lim_y = (lim[0][1], lim[1][1])
        self._dx = abs(self._lim_x[1] - self._lim_x[0]) # Maybe will be used for scaling vectors
        self._dy = abs(self._lim_y[1] - self._lim_y[0])
        self._ax = ax
        self._scale = scale

    def play(self,
             t0:float=0,
             tf:float=10,
             dt:float=0.03,
             ax=None,
             own_ships_verbose=['enveloppe', 'frame', 'acceleration', 'velocity', 'forces'],
             target_ships_verbose=['enveloppe'],
             **kwargs
             ):
        """
        Play the environment during an interval of time.
        """
        if ax is None:
            _, ax = plt.subplots()

        t = t0
        while True:
            loop_start = time.time()
            ax.cla()
            ax.set_xlim(*self._lim_x)
            ax.set_ylim(*self._lim_y)
            self._env.step()
            self._env.plot(t, self._lim, own_ship_physics=own_ships_verbose, target_ship_physics=target_ships_verbose, ax=ax)
            ax.set_title(f"t = {t:.2f}")
            t += dt

            if t > tf:
                ax.set_title(f"t = {tf:.2f} : Done")
                plt.waitforbuttonpress(120)
                break

            loop_end = time.time()
            # print(f"{t:.2f} | Loop time: {loop_end - loop_start}")
            plt.pause(max(1e-9, dt - (loop_end - loop_start)))

    @property
    def lim_x(self) -> tuple:
        return self._lim_x
    
    @property
    def lim_y(self) -> tuple:
        return self._lim_y

            

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
