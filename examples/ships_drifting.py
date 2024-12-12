
if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    from nav_env.viz.matplotlib_screen import MatplotlibScreen as Screen
    # from nav_env.viz.pygame_screen import PyGameScreen as Screen
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.ships.ship import *
    from nav_env.ships.states import *
    from nav_env.wind.wind_source import UniformWindSource
    from nav_env.obstacles.obstacles import *
    from nav_env.ships.sailing_ship import SailingShip
    from nav_env.risk.monitor import RiskMonitor


    # Simulation parameters
    lim = 300
    xlim, ylim = (-lim, -lim), (lim, lim)
    dt = 0.1
    tf = 60

    # Shore (Made of obstacles)
    obs1 = Circle(0, 40, 50, id=1)
    obs2 = Ellipse(-50, -50, 100, 20, id=2)
    obs3 = Obstacle(xy=[(0, 0), (50, 0), (80, 10), (100, 50), (60, 90), (10, 30)], id=3).rotate_and_translate(200, 30, 90)

    # Ostacles (Moving)
    kin_obs = ObstacleWithKinematics(initial_state=States3(-200., -100., 0., 10., -10., 30.), xy=[(0, 0), (50, 0), (80, 10), (100, 50), (60, 90), (10, 30)], id=4)
    sailing_ship = SailingShip(length=200, width=50, ratio=7/9, initial_state=States2(100., -100., -10., 5.), id=5, domain_margin_wrt_enveloppe=50)

    # Ships
    ship1 = Ship(States3(-150., -200., 180., 20., 30., 10.), integrator=Euler(dt), name="Ship1", domain=Ellipse(0, 0, 50, 100))
    ship2 = Ship(States3(-50., 50., -70., 10., 0., -10.), integrator=Euler(dt), name="Ship2", domain_margin_wrt_enveloppe=30)
    ship3 = Ship(States3(10., -100., -30., 0., 0., 0.), integrator=Euler(dt), name="Ship3", domain=Circle(0, 0, 50))
    ship4 = Ship(States3(250., -200., 0., 0., 0., 60.), integrator=Euler(dt), name="Ship4")
    ship5 = Ship(States3(250., 250., 80., -100., -100., 10.), integrator=Euler(dt), name="Ship5")

    # Wind
    uniform_wind = UniformWindSource(-30, 30)

    # Environment
    env = Env(
        own_ships=[ship1, ship2, ship3, ship4],
        target_ships=[ship5],
        wind_source=uniform_wind,
        obstacles=[sailing_ship],
        shore=[obs1, obs2, obs3]
        )

    from nav_env.risk.ttg import TTG
    from nav_env.risk.distance import Distance

    # Screen to display simulation results
    screen = Screen(env, scale=1, lim=(xlim, ylim), monitor=RiskMonitor([TTG, Distance], dt=1))
    screen.play(dt=dt, tf=tf, own_ships_verbose={'enveloppe':1, 'name':1}, target_ships_verbose={'enveloppe':1, 'name':1})  