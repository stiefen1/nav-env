def figure3_distance_vs_ttg() -> None:
    import os, matplotlib.pyplot as plt
    from nav_env.map.map import Map
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.wind.wind_source import UniformWindSource
    from nav_env.ships.ship import Ship
    from nav_env.ships.states import States3
    from nav_env.simulation.integration import Euler
    from nav_env.risk.ttg import TTG
    from nav_env.risk.distance import Distance
    from math import pi 


    dt = 1.
    center = (43150, 6958000)
    size = (1500, 850)
    xlim = center[0]-size[0]/2, center[0]+size[0]/2
    ylim = center[1]-size[1]/2, center[1]+size[1]/2

    # Load map
    config_path = os.path.join('examples', 'config', 'config_seacharts.yaml')
    shore = Map(config_path, depth=0)
    obs = shore.get_obstacle_collection_in_window_from_enc(center=center, size=size)

    ship1 = Ship(States3(43000, 6957670, -10., 3., 10., 0.), integrator=Euler(dt), name="Ship1")
    ship2 = Ship(States3(43080, 6958309, -135., 5., -5., 0.), integrator=Euler(dt), name="Ship2")
    # ship3 = Ship(States3(43040, 6957925, 50., -3., 10., 0.), integrator=Euler(dt), name="Ship3")

    # Wind
    uniform_wind = UniformWindSource(10, -10)

    # Environment
    env = Env(
        own_ships=[ship1, ship2],
        wind_source=uniform_wind,
        shore=obs.as_list()
        )
    
    t_max = 200
    ttg = TTG(env)
    distance = Distance(env)

    # Scenario 1
    ttg1 = ttg.calculate(ship1, t_max=t_max)
    d1 = distance.calculate(ship1)

    states1 = [p[1] for p in ttg.full_trajectory]
    x1 = [s.x for s in states1]
    y1 = [s.y for s in states1]

    # Scenario 2
    ttg2 = ttg.calculate(ship2, t_max=t_max)
    d2 = distance.calculate(ship2)

    states2 = [p[1] for p in ttg.full_trajectory]
    x2 = [s.x for s in states2]
    y2 = [s.y for s in states2]

    # Print results
    print(f"d1: {d1} | TTG1: {ttg1}")
    print(f"d2: {d2} | TTG2: {ttg2}")

    # Plot results
    ## Shore
    ax = obs.plot()

    ## Ships
    ship1.plot(ax=ax, c='r')
    ship2.plot(ax=ax, c='b')
    
    ## Trajectories
    ax.plot(x1, y1, '--r', label=f"Traj. (ship 1) after LOPP")
    ax.plot(x2, y2, '--b', label=f"Traj. (ship 2) after LOPP")

    ## Wind
    uniform_wind.quiver(ax=ax, nx=5, ny=5, lim=((xlim[0], ylim[0]), (xlim[1], ylim[1])), alpha=0.5, color='orange', label='Wind (uniform)')

    # Risk metric values
    bbox_params = {'facecolor':'white', 'alpha':1, 'linewidth':0.1}
    
    ## Ship 1
    text_d1 = ax.text(x1[0]-340, y1[0], "$d^{min}_1$" + f"={d1:.0f} [m]", color='r')
    text_ttg1 = ax.text(x1[-1]+38, y1[-1]-33, "$TTG_1$" + f"={int(ttg1)} [s]", color='r')
    text_d1.set_bbox(bbox_params)
    text_ttg1.set_bbox(bbox_params)

    ## Ship 2
    text_d2 = ax.text(x2[0]-340, y2[0]+30, "$d^{min}_2$" + f"={d2:.0f} [m]", color='blue')
    text_ttg2 = ax.text(x2[-1]-30, y2[-1]-80, "$TTG_2$" + f"={int(ttg2)} [s]", color='blue')
    text_d2.set_bbox(bbox_params)
    text_ttg2.set_bbox(bbox_params)

    ## Time to shine
    ax.legend(loc='center left', framealpha=1)
    ax.locator_params(axis='y', nbins=5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()


if __name__=="__main__":
    figure3_distance_vs_ttg()