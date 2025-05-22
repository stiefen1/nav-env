from nav_env.obstacles.obstacles import MovingObstacle
from nav_env.risk.utils import get_relative_position_and_speed
from nav_env.risk.risk import RiskMetric
from nav_env.environment.environment import NavigationEnvironment
import numpy as np
from nav_env.ships.sailing_ship import SailingShip
from nav_env.ships.states import States3
from scipy.stats import norm


def get_tcpa(own_ship: MovingObstacle, target_ship: MovingObstacle) -> float:
    """
    Calculate the Degree of Domain Violation of the target ship by the own ship.
    """
    px_rel, py_rel, vx_rel, vy_rel = get_relative_position_and_speed(own_ship, target_ship)
    
    # Norm of relative speed vector
    v_rel_squared = (vx_rel**2 + vy_rel**2)
    if v_rel_squared == 0:
        return -float("inf")

    tcpa:float = -(px_rel * vx_rel + py_rel * vy_rel)/(v_rel_squared)
    return tcpa

def get_dcpa(own_ship: MovingObstacle, target_ship: MovingObstacle) -> float:
    """
    Calculate the Degree of Domain Violation of the target ship by the own ship.
    """
    px_rel, py_rel, vx_rel, vy_rel = get_relative_position_and_speed(own_ship, target_ship)
    
    # Norm of relative speed vector
    v_rel_squared = (vx_rel**2 + vy_rel**2)
    if v_rel_squared == 0: # if OS and TS have exact the same speed vectors then DCPA is undefined (0/0)
        return float("NaN")

    dcpa:float = abs( (px_rel * vy_rel - py_rel * vx_rel) / (v_rel_squared**0.5) )
    return dcpa

class TCPA(RiskMetric):
    def __init__(self, env:NavigationEnvironment):
        super().__init__(env)
        
    def calculate(self, ship:MovingObstacle) -> float:
        """
        Calculate the Time to Closest Point of Approach of the target ship by the own ship.
        """
        min_tcpa = float("inf")
        for target_ship in self._env.target_ships:
            tcpa = get_tcpa(ship, target_ship)
            if tcpa < min_tcpa:
                min_tcpa = tcpa

        for moving_obstacle in self._env.obstacles:
            tcpa = get_tcpa(ship, moving_obstacle)
            if tcpa < min_tcpa:
                min_tcpa = tcpa
        return min_tcpa
    
    def plot(self, ax=None, **kwargs):
        pass

class DCPA(RiskMetric):
    def __init__(self, env:NavigationEnvironment):
        super().__init__(env)
        
    def calculate(self, ship:MovingObstacle) -> float:
        """
        Calculate the Distance to Closest Point of Approach of the target ship by the own ship.
        """
        min_dcpa = float("inf")
        for target_ship in self._env.target_ships:
            dcpa = get_dcpa(ship, target_ship)
            if dcpa < min_dcpa:
                min_dcpa = dcpa

        for moving_obstacle in self._env.obstacles:
            dcpa = get_dcpa(ship, moving_obstacle)
            if dcpa < min_dcpa:
                min_dcpa = dcpa
        return min_dcpa
    
    def plot(self, ax=None, **kwargs):
        pass


def test() -> None:
    from nav_env.ships.sailing_ship import SailingShip
    from nav_env.ships.states import States3
    import matplotlib.pyplot as plt
    import numpy as np

    os = SailingShip(initial_state=States3(x_dot=0, y_dot=4))
    ts = SailingShip(initial_state=States3(x=20, y=50, x_dot=-5, y_dot=-5))
    
    DCPA = get_dcpa(os, ts)
    TCPA = get_tcpa(os, ts)
    # print(get_dcpa(os, ts), get_tcpa(os, ts))
    ax = os(1e-6).plot()
    ts(1e-6).plot(ax=ax)
    ax.set_aspect('equal')
    plt.show(block=False)
    ylim = ax.get_ylim()
    xlim = [-ylim[1]/2, ylim[1]/2]

    tf = 10
    for ti in np.linspace(0, tf, 100):
        ax.cla()
        os(ti+1e-9).plot(ax=ax)
        ts(ti+1e-9).plot(ax=ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f"TCPA: {TCPA-ti:.2f}")

        plt.pause(tf/100)

def show_tcpa_in_heading_and_speed_space() -> None:
    from nav_env.ships.sailing_ship import SailingShip
    from nav_env.ships.states import States3
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    os = SailingShip(initial_state=States3(x_dot=0, y_dot=4))

    Nh, Nv = 50, 50
    headings = np.linspace(-45, 45, Nh)
    speeds = np.linspace(1, 10, Nv)
    tcpa = np.zeros((Nh, Nv, 3))
    dcpa = np.zeros((Nh, Nv, 3))

    for i, heading in enumerate(headings):
        for j, v in enumerate(speeds):
            # Convert heading and speed into vx, vy
            heading_rad = heading*np.pi/180.0
            vy = v * np.cos(heading_rad)
            vx = -v * np.sin(heading_rad)
            ts = SailingShip(initial_state=States3(x=20, y=50, x_dot=vx, y_dot=vy))

            # Compute DCPA % TCPA
            tcpa[i, j] = np.array([heading, v, get_tcpa(os, ts)])
            dcpa[i, j] = np.array([heading, v, get_dcpa(os, ts)])
    

    MARKER_SIZE = 10
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    ax.scatter(dcpa[:, :, 0], dcpa[:, :, 1], dcpa[:, :, 2], c=dcpa[:, :, 2], s=MARKER_SIZE)
    ax.set_xlabel("Headings")
    ax.set_ylabel("Speed")
    ax.set_zlabel("DCPA")
    # Noisy measurement
    mu_h, sigma_h = -5, 5*2
    mu_v, sigma_v = 3.8, 0.5*2

    # Nominal (naive) case
    # mu_h_rad = mu_h*np.pi/180.0
    # vy = mu_v * np.cos(mu_h_rad)
    # vx = -mu_v * np.sin(mu_h_rad)
    # ts = SailingShip(initial_state=States3(x=20, y=50, x_dot=vx, y_dot=vy))
    # tcpa_nom = get_tcpa(os, ts)
    # dcpa_nom = get_dcpa(os, ts)

    # ax.scatter(mu_h, mu_v, dcpa_nom, c='red', s=100)
    

    # dcpa_particles = get_dcpa_from_particle_filter(os, mu_h=mu_h, sigma_h=sigma_h, mu_v=mu_v, sigma_v=sigma_v, n_particles=50)
    dcpa_particles = np.zeros((Nh, Nv, 3))
    for i, heading in enumerate(headings):
        for j, v in enumerate(speeds):
            # Compute DCPA using from particle filter
            dcpa_particles[i, j] = np.array([heading, v, get_dcpa_from_particle_filter(os, mu_h=heading, sigma_h=sigma_h, mu_v=v, sigma_v=sigma_v, n_particles=50)])
        print(f"{(i+1)*Nv}/{Nh*Nv}")

    # ax.scatter(mu_h, mu_v, dcpa_particles, c='green', s=100)
    # print(dcpa_particles, dcpa_nom)

    fig2 = plt.figure()
    ax = fig2.add_subplot(projection='3d')
    ax.scatter(dcpa_particles[:, :, 0], dcpa_particles[:, :, 1], dcpa_particles[:, :, 2], c=dcpa_particles[:, :, 2], s=MARKER_SIZE)
    ax.set_xlabel("Headings")
    ax.set_ylabel("Speed")
    ax.set_zlabel("Filtered DCPA")
    

    fig3 = plt.figure()
    ax = fig3.add_subplot(projection='3d')
    ax.scatter(dcpa_particles[:, :, 0], dcpa_particles[:, :, 1], dcpa[:, :, 2] - dcpa_particles[:, :, 2], c=dcpa[:, :, 2] - dcpa_particles[:, :, 2], s=MARKER_SIZE)
    ax.set_xlabel("Headings")
    ax.set_ylabel("Speed")
    ax.set_zlabel("Expected over-estimation")

    plt.show()

def show_dcpa_prob_in_heading_and_speed_space() -> None:
    from nav_env.ships.sailing_ship import SailingShip
    from nav_env.ships.states import States3
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    os = SailingShip(initial_state=States3(x_dot=0, y_dot=4))

    Nh, Nv = 30, 30
    headings = np.linspace(-45, 45, Nh)
    speeds = np.linspace(1, 10, Nv)
    tcpa = np.zeros((Nh, Nv, 3))
    dcpa = np.zeros((Nh, Nv, 3))

    for i, heading in enumerate(headings):
        for j, v in enumerate(speeds):
            # Convert heading and speed into vx, vy
            heading_rad = heading*np.pi/180.0
            vy = v * np.cos(heading_rad)
            vx = -v * np.sin(heading_rad)
            ts = SailingShip(initial_state=States3(x=20, y=50, x_dot=vx, y_dot=vy))

            # Compute DCPA % TCPA
            tcpa[i, j] = np.array([heading, v, get_tcpa(os, ts)])
            dcpa[i, j] = np.array([heading, v, get_dcpa(os, ts)])
    

    MARKER_SIZE = 10
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    ax.scatter(dcpa[:, :, 0], dcpa[:, :, 1], dcpa[:, :, 2], c=dcpa[:, :, 2], s=MARKER_SIZE)
    ax.set_xlabel("Headings")
    ax.set_ylabel("Speed")
    ax.set_zlabel("DCPA")
    # Noisy measurement
    mu_h, sigma_h = -5, 5*2
    mu_v, sigma_v = 3.8, 0.5*2
   

    # dcpa_particles = get_dcpa_from_particle_filter(os, mu_h=mu_h, sigma_h=sigma_h, mu_v=mu_v, sigma_v=sigma_v, n_particles=50)
    THRESHOLD = 40
    dcpa_particles = np.zeros((Nh, Nv, 3))
    for i, heading in enumerate(headings):
        for j, v in enumerate(speeds):
            # Compute DCPA using from particle filter
            dcpa_particles[i, j] = np.array([heading, v, get_prob_that_dcpa_is_higher_than(THRESHOLD, os, mu_h=heading, sigma_h=sigma_h, mu_v=v, sigma_v=sigma_v, n_particles=50)])
        print(f"{(i+1)*Nv}/{Nh*Nv}")


    fig2 = plt.figure()
    ax = fig2.add_subplot(projection='3d')
    ax.scatter(dcpa_particles[:, :, 0], dcpa_particles[:, :, 1], dcpa_particles[:, :, 2], c=dcpa_particles[:, :, 2], s=MARKER_SIZE)
    ax.set_xlabel("Headings")
    ax.set_ylabel("Speed")
    ax.set_zlabel(f"Probability that dcpa >= {THRESHOLD}")
    

    plt.show()
    
def get_prob_that_dcpa_is_higher_than(threshold, os, mu_h, sigma_h, mu_v, sigma_v, n_particles:int=50):
    n_above_threshold = 0
    for i in range(n_particles):
        hi = np.random.normal(mu_h, sigma_h)
        vi = np.random.normal(mu_v, sigma_v)
        hi_rad = hi*np.pi/180.0
        vy = vi * np.cos(hi_rad)
        vx = -vi * np.sin(hi_rad)
        ts = SailingShip(initial_state=States3(x=20, y=50, x_dot=vx, y_dot=vy))
        dcpa_i = get_dcpa(os, ts)

        if dcpa_i > threshold:
            n_above_threshold += 1
    
    # dcpa_particles = dcpa_particles / sum_of_likelihood
    return n_above_threshold / n_particles

def get_dcpa_from_particle_filter(os, mu_h, sigma_h, mu_v, sigma_v, n_particles:int=50):
    dcpa_particles = 0
    # sum_of_likelihood = 0
    for i in range(n_particles):
        hi = np.random.normal(mu_h, sigma_h)
        vi = np.random.normal(mu_v, sigma_v)
        hi_rad = hi*np.pi/180.0
        vy = vi * np.cos(hi_rad)
        vx = -vi * np.sin(hi_rad)
        ts = SailingShip(initial_state=States3(x=20, y=50, x_dot=vx, y_dot=vy))
        dcpa_i = get_dcpa(os, ts)

        # NO NEED TO DO THAT BECAUSE I ALREADY SAMPLE hi, vi FROM THEIR NORMAL DISTRIBUTION
        # I'D HAVE TO DO THIS ONLY IF I SAMPLE FROM A UNIFORM DISTRIBUTION -> "IMPORTANCE WEIGHTS"
        # hi_likelihood = norm.pdf(hi, loc=mu_h, scale=sigma_h)
        # vi_likelihood = norm.pdf(vi, loc=mu_v, scale=sigma_v)
        # likelihood = hi_likelihood*vi_likelihood
        likelihood = 1/n_particles
        dcpa_particles += (likelihood*dcpa_i)
        # sum_of_likelihood += likelihood
    
    # dcpa_particles = dcpa_particles / sum_of_likelihood
    return dcpa_particles



if __name__ == "__main__":
    # test()
    # show_tcpa_in_heading_and_speed_space()
    show_dcpa_prob_in_heading_and_speed_space()