from nav_env.control.guidance import GuidanceBase, Guidance
from nav_env.control.navigation import NavigationBase, Navigation
from nav_env.control.controller import ControllerBase, Controller
from nav_env.control.command import Command
from nav_env.ships.states import States3


class GNC:
    def __init__(self,
                 guidance:GuidanceBase,
                 navigation:NavigationBase,
                 controller:ControllerBase,
                 *args,
                 **kwargs
                 ):
        self._guidance = guidance or Guidance()
        self._navigation = navigation or Navigation()
        self._controller = controller or Controller()

    def get(self, ship) -> Command:
        observed_state:States3 = self._navigation.observe(ship) # State is in ship frame
        desired_state, info = self._guidance.get(observed_state)
        return self._controller.get(observed_state, desired_state, **info) # e.g. info can contain initial_guess for NMPC
    
    
    def reset(self) -> None:
        self._guidance.reset()
        self._navigation.reset()
        self._controller.reset()

def test() -> None:
    from nav_env.ships.ship import Ship
    from nav_env.control.PID import HeadingAndSpeedController
    from nav_env.control.LOS import LOSLookAhead
    import matplotlib.pyplot as plt
    from nav_env.environment.environment import NavigationEnvironment
    import numpy as np

    wpts = [
        (0, 0),
        (200, 400),
        (480, 560),
        (900, 600),
        (1250, 950),
        (1500, 1500)
    ]

    # wpts = [
    #     (0, 0),
    #     (100, 1300),
    #     (1200, 1500),
    #     (1000, 100),
    #     (200, 100)
    # ]

    fig = plt.figure()
    ax = fig.add_subplot()
    for wpt in wpts:
        ax.scatter(*wpt, c='black')
    plt.show()

    dt = 1
    name="os"

    ship = Ship(
        states=States3(0, 50, x_dot=3, y_dot=3, psi_deg=-45),
        guidance=LOSLookAhead(
            waypoints=wpts,
            radius_of_acceptance=100.,
            current_wpt_idx=1,
            kp=3e-4, # 7e-3
            desired_speed=4.
        ),
        controller=HeadingAndSpeedController(
            pid_gains_heading=(-5e5, 0, -5e6),
            pid_gains_speed=(8e4, 1e4, 0),
            dt=dt
        ),
        name=name
    )
    
    env = NavigationEnvironment(
        own_ships=[ship],
        dt=dt
    )


    lim = ((-20, -20), (1800, 1800))
    ax = env.plot(lim)
    plt.show(block=False)
    x, y = [], []

    tf = 5000
    for t in np.linspace(0, tf, int(tf//dt)):
        ax.cla()
        for wpt in wpts:
            ax.scatter(*wpt, c='black')
        ax.scatter(*ship._gnc._guidance.current_waypoint, c='red')
        ax.set_title(f"{t:.2f}")
        env.step()
        v = np.linalg.norm(ship.states.xy_dot)
        print(v)
        if t%10 > 0:
            x.append(ship.states.x)
            y.append(ship.states.y)
        ax.plot(x, y, '--r')
        env.plot(lim, ax=ax)
        plt.pause(1e-9)

    plt.pause()
    
    

if __name__=="__main__":
    test()
        