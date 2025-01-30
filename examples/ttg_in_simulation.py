def test():
    from nav_env.simulation.simulator import Simulator, SimulationRecord
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.ships.ship import SimpleShip
    from nav_env.ships.states import States3

    from nav_env.risk.ttg import TTG, TTGMaxWorsening
    from nav_env.obstacles.obstacles import Obstacle, Ellipse
    from nav_env.risk.monitor import RiskMonitor
    from nav_env.wind.wind_source import UniformWindSource
    import matplotlib.pyplot as plt, os
    from math import cos, sin, pi

    # Make folder
    record_folder = 'recordings'
    filename = 'test.csv'
    path_to_file = os.path.join(record_folder, filename)
    os.makedirs(record_folder, exist_ok=True)

    # Shore (Made of obstacles)
    island1 = Obstacle(xy=[(-400, -300), (-200, -300), (-100, -150), (-150, -50), (0, 200), (-400, 300)]).buffer(-100).buffer(50).rotate(-10).translate(0, 50)
    island2 = Obstacle(xy=[(200, -300), (400, -300), (300, -150), (250, -50), (200, 200), (0, 300)]).buffer(-50).buffer(25).rotate(10).translate(0, -150)
    
    # Ships
    os1 = SimpleShip(States3(-450., -900., -20., 7.5, 18., 0.), name="OS")

    # Wind
    angle = 330
    wind = UniformWindSource(-30*sin(angle*pi/180.), 30*cos(angle*pi/180.))

    # Environment
    env = Env(
        shore=[island1, island2],
        own_ships=[os1],
        wind_source=wind,
        )
    
    # Monitor
    monitor = RiskMonitor([TTG])
    sim = Simulator(env=env, monitor=monitor)
    sim.run(tf=100, record_results_dt=5)

    # Save simulation record
    sim.record.save(path_to_file)

    # Load it back
    new_sim_record = SimulationRecord(path_to_existing_data=path_to_file)

    # Plot environment
    lim = 800
    ax = env.plot(lim=((-lim, -lim), (lim, lim)))

    ax.scatter(new_sim_record['OS']['states']['x'], new_sim_record['OS']['states']['y'], c=new_sim_record['OS']['risks']['TTG'])
    ax.axis('equal')
    plt.show()


if __name__=="__main__":
    test()