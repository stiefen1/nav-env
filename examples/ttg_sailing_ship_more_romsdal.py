def test() -> None:
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
    from nav_env.risk.ttg import TTG, TTGMaxWorsening, TTGWorstCase, TTGExpectation, TTGCurvature
    from nav_env.risk.monitor import RiskMonitor

    dt = 1.
    tf = 160
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
        (0, States3(43080, 6957615)),
        (40, States3(43025, 6957906)),
        (80, States3(43234, 6958050)),
        (120, States3(43610, 6958251)),
        (160, States3(43425, 6958354))
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


    # Screen to display simulation results
    screen = Screen(env, scale=1, lim=((xlim[0], ylim[0]), (xlim[1], ylim[1])), monitor=RiskMonitor([TTG, TTGCurvature], dt=1))
    screen.play(dt=dt, tf=tf, own_ships_verbose={'enveloppe':1, 'name':1})  

    # plt.show()

if __name__=="__main__":
    test()