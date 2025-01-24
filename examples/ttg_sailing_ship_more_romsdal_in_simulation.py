import os
from nav_env.map.map import Map
from nav_env.environment.environment import NavigationEnvironment as Env
from nav_env.wind.wind_source import UniformWindSource
from nav_env.ships.ship import Ship
from nav_env.ships.sailing_ship import SailingShip
from nav_env.control.path import TimeStampedWaypoints
from nav_env.ships.states import States3
from nav_env.simulation.integration import Euler
from nav_env.viz.matplotlib_screen import MatplotlibScreen as Screen
from nav_env.risk.ttg import TTG, TTGMaxWorsening, TTGWorstCase, TTGExpectation, TTGCurvature, TTGExpectedWorsening
from nav_env.risk.monitor import RiskMonitor
from nav_env.simulation.simulator import Simulator, SimulationRecord
import matplotlib.pyplot as plt

def test() -> None:


    dt = 1.
    tf = 240
    center = (43150, 6958000)
    size = (1500, 850)
    xlim = center[0]-size[0]/2, center[0]+size[0]/2
    ylim = center[1]-size[1]/2, center[1]+size[1]/2

    # Load map
    config_path = os.path.join('examples', 'config', 'config_seacharts.yaml')
    shore = Map(config_path, depth=0)
    obs = shore.get_obstacle_collection_in_window_from_enc(center=center, size=size)

    # Route to follow
    wpts = [
        (0, States3(43900, 6957472)),
        (40, States3(43642, 6957472)),
        (80, States3(43080, 6957615)),
        (120, States3(43025, 6957906)),
        (160, States3(43234, 6958050)),
        (200, States3(43610, 6958251)),
        (240, States3(43425, 6958354))
        ]

    # ship1 = Ship(States3(*center, -40., 10., 10., 0.), integrator=Euler(dt), name="Ship1")
    ship2 = SailingShip(length=40, width=10, pose_fn=TimeStampedWaypoints(wpts))

    # Wind
    uniform_wind = UniformWindSource(10, -30)

    # Environment
    env = Env(
        own_ships=[ship2],
        wind_source=uniform_wind,
        shore=obs.as_list()
        )


    sim = Simulator(env=env, monitor=RiskMonitor([TTG, TTGWorstCase]))
    sim.run(tf=tf, record_results_dt=5, dt=dt)

    # # Save simulation record
    filename = 'more_og_romsdal_worst_case'
    path_to_filename = 'recordings\\' + filename + '.csv'
    sim.record.save(path_to_filename)

    # Load it back
    new_sim_record = SimulationRecord(path_to_existing_data=path_to_filename)

    # Plot environment
    ax = env.plot(lim=((xlim[0], ylim[0]), (xlim[1], ylim[1])))

    ax.scatter(new_sim_record['SailingShip']['states']['x'], new_sim_record['SailingShip']['states']['y'], c=new_sim_record['SailingShip']['risks']['TTGWorstCase'])
    ax.axis('equal')
    plt.show()
    # plt.show()


if __name__=="__main__":
    test()