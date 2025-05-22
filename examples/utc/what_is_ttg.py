def figure3_distance_vs_ttg() -> None:
    import os, matplotlib.pyplot as plt
    from nav_env.map.map import Map
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.wind.wind_source import UniformWindSource
    from nav_env.ships.ship import Ship
    from nav_env.ships.sailing_ship import SailingShip
    from nav_env.ships.states import States3
    from nav_env.simulation.integration import Euler
    from nav_env.risk.ttg import TTG
    from nav_env.risk.distance import Distance
    from math import pi 
    from nav_env.control.path import TimeStampedWaypoints


    dt = 1.
    center = (43150, 6958000)
    # size = (1500, 850)
    size = (1000, 850)
    xlim = center[0]-size[0]/2, center[0]+size[0]/2
    ylim = center[1]-size[1]/2, center[1]+size[1]/2

    # Load map
    config_path = os.path.join('examples', 'config', 'more_og_romsdal.yaml')
    shore = Map(config_path, depth=0)
    obs = shore.get_obstacle_collection_in_window_from_enc(center=center, size=size)

    wpts = [
        (0, States3(x=43000, y=6957670)),
        (20, States3(x=43025, y=6957776)),
        (40, States3(x=43083, y=6957891)),
        (60, States3(x=43188, y=6957998)),
        (80, States3(x=43314, y=6958102)),
        (100, States3(x=43469, y=6958172)),
        (120, States3(x=43610, y=6958242))
    ]

    traj = TimeStampedWaypoints(wpts)

    # print("TIMESTAMPED WAYPOINTS: ", traj._timestamped_waypoints)
    # ship1 = Ship(States3(43000, 6957670, -10., 3., 10., 0.), integrator=Euler(dt), name="Ship1", width=20, length=40)
    sailing_ship = SailingShip(pose_fn=traj, width=10, length=25)
    # print("Pose at 10: ", sailing_ship.pose_fn(10))

    _, ax = plt.subplots(figsize=(10, 8.5))

    for t in range(0, 80):
        ship = Ship(states=sailing_ship.pose_fn(t), width=sailing_ship.width, length=sailing_ship.length)

        # Wind
        uniform_wind = UniformWindSource(-7.25, -7.3)

        # Environment
        env = Env(
            own_ships=[ship],
            wind_source=uniform_wind,
            shore=obs.as_list()
            )

        # Plot results
        ## Shore
        ax.cla()
        obs.plot(c='green', ax=ax)
        ax.fill_between([42000, 44000], [6_957_500, 6_957_500], [6_958_500, 6_958_500], color='lightblue')
        obs.fill(ax=ax, c='lightgreen')
        

        # traj.scatter(c='r', marker='x', ax=ax)
        for wpt in wpts:
            ax.scatter(*wpt[1].xy, c='r', marker='x')

        ## Ships
        ship.plot(ax=ax, c='r')
        ship.fill(ax=ax, c='r')
        # sailing_ship(t).plot(ax=ax, c='purple')
        # sailing_ship(t).fill(ax=ax, c='purple')

        t_max = 400
        ttg = TTG(env)
        # print("START TTG")
        ttg_val = ttg.calculate(ship, t_max=t_max, precision_sec=0.1)
        ship_drifting = SailingShip(pose_fn=ttg.timestamped_waypoints, length=ship.length, width=ship.width, fix_heading=False)
        ttg.timestamped_waypoints.plot('--k', ax=ax)
        ship_drifting(ttg_val).plot(ax=ax, c='black')
        ship_drifting(ttg_val).fill(ax=ax, c='black')

        ## Wind
        uniform_wind.compass(((xlim[0], ylim[0]), (xlim[1], ylim[1])), ax=ax, loc='upper left', label='kn')

        ## Time to shine
        # ax.legend(loc='center left', framealpha=1)
        ax.locator_params(axis='y', nbins=5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        # plt.savefig(f"what_is_ttg_{t}.png")
        plt.pause(1e-9)
        # plt.show(block=False)


    


if __name__=="__main__":
    figure3_distance_vs_ttg()