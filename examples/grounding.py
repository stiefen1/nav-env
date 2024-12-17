if __name__ == "__main__":
    from nav_env.ships.ship import *
    from nav_env.ships.states import *
    from nav_env.obstacles.obstacles import *
    from nav_env.wind.wind_source import UniformWindSource
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.viz.matplotlib_screen import MatplotlibScreen as Screen
    from nav_env.risk.monitor import RiskMonitor
    from nav_env.risk.ttg import TTG, TTGMaxWorsening

    # Simulation parameters
    lim = 600
    xlim, ylim = (-lim, -lim), (lim, lim)
    dt = 0.1
    tf = 60

    # Shore (Made of obstacles)
    island1 = Obstacle(xy=[(-400, -300), (-200, -300), (-100, -150), (-150, -50), (0, 200), (-400, 300)]).buffer(-100).buffer(50).rotate(-10).translate(0, 50)
    island2 = Obstacle(xy=[(200, -300), (400, -300), (300, -150), (250, -50), (200, 200), (0, 300)]).buffer(-50).buffer(25).rotate(10).translate(0, -150)
    
    # Ships
    os1 = SimpleShip(States3(-250., -500., -20., 7.5, 18., 0.), integrator=Euler(dt), domain=Ellipse(0, 0, 200, 100), name="OS1")

    # Wind
    wind = UniformWindSource(-10, -30)

    # Environment
    env = Env(
        shore=[island1, island2],
        own_ships=[os1],#, os2],
        wind_source=wind
        )
    
    # Monitor
    monitor = RiskMonitor([TTG, TTGMaxWorsening], dt=1)
    
    # Screen
    screen = Screen(env, scale=1, lim=(xlim, ylim), monitor=monitor)
    screen.play(tf=tf, dt=dt, own_ships_verbose={'enveloppe':1, 'name':1}, target_ships_verbose={'enveloppe':1, 'name':1, 'domain':1})
