def test():
    from nav_env.simulation.simulator import Simulator, SimulationRecord
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.ships.ship import SimpleShip
    from nav_env.ships.states import States3

    from nav_env.risk.ttg import TTG, TTGMaxWorsening
    from nav_env.obstacles.obstacles import Obstacle, Ellipse
    from nav_env.risk.monitor import RiskMonitor
    from nav_env.wind.wind_source import UniformWindSource
    import matplotlib.pyplot as plt


    # Load it back
    new_sim_record = SimulationRecord(path_to_existing_data='recordings\\ttg_330_deg.csv')

    _, ax = plt.subplots()

    ax.plot(new_sim_record.times, new_sim_record['OS']['risks']['TTG'], '--')
    ax.scatter(new_sim_record.times, new_sim_record['OS']['risks']['TTG'])
    ax.plot(new_sim_record.times, new_sim_record['OS']['risks']['TTGMaxWorsening'], '--')
    ax.scatter(new_sim_record.times, new_sim_record['OS']['risks']['TTGMaxWorsening'])
    ax.legend(['TTG', None, 'TTGMaxWorsening', None])
    ax.grid()
    ax.axis('equal')
    plt.show()


if __name__=="__main__":
    test()