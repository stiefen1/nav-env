from nav_env.ships.states import *

def get_pose(t:float) -> States3:
    return States3(10*t, 10*t)

if __name__ == "__main__":
    from nav_env.ships.ship import *
    from nav_env.obstacles.obstacles import *
    from nav_env.ships.sailing_ship import SailingShip
    from nav_env.risk.monitor import RiskMonitor
    from nav_env.risk.cpa import DCPA, TCPA
    from nav_env.risk.ddv import DDV, TDV, DDV2
    from nav_env.risk.distance import Distance
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.viz.matplotlib_screen import MatplotlibScreen as Screen

    # Simulation parameters
    lim = 300
    xlim, ylim = (-lim, -lim), (lim, lim)
    dt = 0.1
    tf = 30

    # Shore (Made of obstacles)
    island = Obstacle(xy=[(0, 0), (50, 0), (80, 10), (100, 50), (60, 90), (10, 30)]).buffer(100).rotate_and_translate(100, 100, 45)
    
    # Ships
    os1 = SimpleShip(States3(200., -250., 45., -10., 12., 0.), integrator=Euler(dt), domain=Ellipse(100, 0, 200, 100), name="OS1")
    os2 = SimpleShip(States3(-200., -150., -30., 10., 18., 0.), integrator=Euler(dt), domain_margin_wrt_enveloppe=30, name="OS2")

    ts1 = SimpleShip(States3(-150, 150, -150, 10, -15, 0), integrator=Euler(dt), domain=Ellipse(0, 0, 200, 100), name="TS1")
    sailing_ship = SailingShip(20, 10, pose_fn=get_pose, domain=Ellipse(0, 0, 50, 30))

    # Environment
    env = Env(
        shore=[island],
        own_ships=[os1, sailing_ship],#, os2],
        target_ships=[ts1],#, ts2, ts3],
        # obstacles=[ts1, ts2, ts3]
        )
    
    # Monitor
    monitor = RiskMonitor([DDV2], dt=0.5)
    
    screen = Screen(env, scale=1, lim=(xlim, ylim), monitor=monitor)
    screen.play(tf=tf, dt=dt, own_ships_verbose={'enveloppe':1, 'name':1, 'domain':1}, target_ships_verbose={'enveloppe':1, 'name':1})
