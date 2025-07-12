def test_path() -> None:
    import matplotlib.pyplot as plt, numpy as np, os
    from nav_env.ships.simplified_physics import SimpleShipPhysics
    import nav_env.ships.physics as phy
    from nav_env.ships.ship import Ship, States3
    from nav_env.control.path import Waypoints as WP
    from nav_env.control.NMPC import NMPCPathTracking
    from nav_env.actuators.actuators import AzimuthThrusterWithSpeed, AzimuthThruster
    from nav_env.control.guidance import PathProgressionAndSpeedGuidance
    from nav_env.environment.environment import NavigationEnvironment
    from nav_env.wind.wind_source import UniformWindSource
    from nav_env.control.LOS import LOSLookAhead
    from nav_env.control.PID import HeadingAndSpeedController
    from math import pi, cos, sin
    from nav_env.control.allocation import NonlinearControlAllocation

    dt = 5.0
    INTERPOLATION = 'linear' # 'bspline'

    R = 1000
    cx, cy = R, 0
    wpts = WP([
        (R*cos(pi)+cx, R*sin(pi)+cy),
        (R*cos(3*pi/4)+cx, R*sin(3*pi/4)+cy),
        (R*cos(pi/2)+cx, R*sin(pi/2)+cy),
        (R*cos(pi/4)+cx, R*sin(pi/4)+cy),
        (R*cos(0)+cx, R*sin(0)+cy),
        (R*cos(-pi/4)+cx, R*sin(-pi/4)+cy),
        (R*cos(-pi/2)+cx, R*sin(-pi/2)+cy),
        (R*cos(-3*pi/4)+cx, R*sin(-3*pi/4)+cy),
        (R*cos(-pi)+cx, R*sin(-pi)+cy)
    ], interp=INTERPOLATION)

    max_rate_rpm = 2 # = 2 * 2pi rad/min = 2 * 2pi / 60 rad/sec
    max_rate_rad_sec = max_rate_rpm * 2 * pi / 60
    actuators_with_speed = [
        AzimuthThrusterWithSpeed(
            (33, 0), 0, (-max_rate_rad_sec, max_rate_rad_sec), (0, 300), dt
        ),
        AzimuthThrusterWithSpeed(
            (-33, 0), 0, (-max_rate_rad_sec, max_rate_rad_sec), (0, 300), dt
        )
    ]
    
    actuators = [
        AzimuthThruster(
            (33, 0), 0, (-180, 180), (0, 300), dt
        ),
        AzimuthThruster(
            (-33, 0), 0, (-180, 180), (0, 300), dt
        )
    ]

    path_to_params = os.path.join('nav_env', 'ships', 'blindheim_risk_2020.json')

    ship1 = Ship(
        states=States3(x=wpts[0][0], y=wpts[0][1], x_dot=0.5, y_dot=2.7, psi_deg=-20),
        physics=phy.ShipPhysics(path_to_params),
        actuators=actuators_with_speed,
        guidance=PathProgressionAndSpeedGuidance(
            wpts, 3
        ),
        controller=NMPCPathTracking(
            route=wpts,
            physics=SimpleShipPhysics(path_to_params),
            actuators=actuators_with_speed,
            weights={
                "kappa": 1*np.array([20, 5e4, 1]).T,
                # "Lambda": 0e-2*np.diag([0, 1]),
                # "Delta": 0e-2*np.diag([1e-2, 1])
                "Lambda": 1*np.diag([5e1, 5e-3, 5e1, 5e-3]), # a_rate, speed, a_rate, speed
                "Delta": 1*np.diag([2e2, 1e5, 2e2, 1e5])
            },
            horizon=40,
            dt=dt
        )
    )

    ship2 = Ship(
        states=States3(x=wpts[0][0], y=wpts[0][1], x_dot=0.5, y_dot=2.7, psi_deg=-20),
        physics=phy.ShipPhysics(path_to_params),
        guidance=LOSLookAhead(
            waypoints=wpts,
            radius_of_acceptance=200.,
            current_wpt_idx=1,
            kp=1e-4, # 7e-3
            desired_speed=3.
        ),
        controller=HeadingAndSpeedController(
            pid_gains_heading=(2.5e5, 0, 1e5),
            pid_gains_speed=(8e4, 1e4, 0),
            dt=dt,
            allocation=NonlinearControlAllocation(actuators=actuators)
        ),
        actuators=actuators,
        # guidance=LOSLookAhead(
        #     waypoints=wpts,
        #     radius_of_acceptance=200.,
        #     current_wpt_idx=1,
        #     kp=1e-8, # 7e-3
        #     desired_speed=3.
        # ),
        # controller=HeadingAndSpeedController(
        #     pid_gains_heading=(2.5e5, 0, 1e5),
        #     pid_gains_speed=(8e4, 1e4, 0),
        #     dt=dt
        # ),
        name="ship2"
    )

    env = NavigationEnvironment(
        own_ships=[ship1, ship2],
        dt=dt,
        wind_source=UniformWindSource(0, 0)
    )




    lim = ((-100, -R-100), (2*R+100, R+100))
    ax = env.plot(lim)
    plt.show(block=False)
    x1, y1 = [], []
    x2, y2 = [], []

    tf = 2000
    for t in np.linspace(0, tf, int(tf//dt)):
        ax.cla()
        wpts.scatter(ax=ax, color='black')
        ax.set_title(f"{t:.2f}")
        env.step()
        if t%10 > 0:
            x1.append(ship1.states.x)
            y1.append(ship1.states.y)
            x2.append(ship2.states.x)
            y2.append(ship2.states.y)
        ax.plot(x1, y1, '--r')
        ax.plot(x2, y2, '--g')
        env.plot(lim, ax=ax, )
        ship1._gnc._controller.plot_trajectory(ax=ax)
        ship1._gnc._controller.plot_desired_trajectory_from_optimization(ax=ax)
        ax.set_aspect('equal')
        print(f"{t:.0f} | GOAL: ", ship2._gnc.goal())
        plt.pause(1e-9)

    plt.close()

    plt.figure()
    plt.title("u(t)")
    plt.plot(ship1.logs["times"], np.array(ship1.logs["states"])[:, 3])
    plt.plot(ship2.logs["times"], np.array(ship2.logs["states"])[:, 3], '--r')
    plt.ylim([0, 5])
    plt.show()
    plt.close()

    plt.figure()
    plt.title("Propeller speed")
    plt.plot(ship1.logs["times"], np.array(ship1.logs["commands"])[:, 1])
    # print(np.array(ship1.logs["commands"]).shape)
    plt.plot(ship1.logs["times"], np.array(ship1.logs["commands"])[:, 3])
    plt.ylim([0, 320])
    plt.show()
    plt.close()

    plt.figure()
    plt.title("Azimuth rate")
    plt.plot(ship1.logs["times"], np.array(ship1.logs["commands"])[:, 0])
    # print(np.array(ship1.logs["commands"]).shape)
    plt.plot(ship1.logs["times"], np.array(ship1.logs["commands"])[:, 2])
    plt.ylim([-1, 1])
    plt.show()
    plt.close()

def test_path_but_with_proper_simulator() -> None:
    import matplotlib.pyplot as plt, numpy as np, os
    from nav_env.ships.simplified_physics import SimpleShipPhysics
    import nav_env.ships.physics as phy
    from nav_env.ships.ship import Ship, States3
    from nav_env.control.path import Waypoints as WP
    from nav_env.control.NMPC import NMPCPathTracking
    from nav_env.actuators.actuators import AzimuthThrusterWithSpeed, AzimuthThruster
    from nav_env.control.guidance import PathProgressionAndSpeedGuidance
    from nav_env.environment.environment import NavigationEnvironment
    from nav_env.wind.wind_source import UniformWindSource
    from nav_env.control.LOS import LOSLookAhead
    from nav_env.control.PID import HeadingAndSpeedController
    from math import pi, cos, sin
    from nav_env.control.allocation import NonlinearControlAllocation
    from nav_env.simulation.simulator import Simulator

    dt = 5.0
    INTERPOLATION = 'linear' # 'bspline'

    R = 1000
    cx, cy = R, 0
    wpts = WP([
        (R*cos(pi)+cx, R*sin(pi)+cy),
        (R*cos(3*pi/4)+cx, R*sin(3*pi/4)+cy),
        (R*cos(pi/2)+cx, R*sin(pi/2)+cy),
        (R*cos(pi/4)+cx, R*sin(pi/4)+cy),
        (R*cos(0)+cx, R*sin(0)+cy),
        (R*cos(-pi/4)+cx, R*sin(-pi/4)+cy),
        (R*cos(-pi/2)+cx, R*sin(-pi/2)+cy),
        (R*cos(-3*pi/4)+cx, R*sin(-3*pi/4)+cy),
        (R*cos(-pi)+cx, R*sin(-pi)+cy)
    ], interp=INTERPOLATION)

    max_rate_rpm = 2 # = 2 * 2pi rad/min = 2 * 2pi / 60 rad/sec
    max_rate_rad_sec = max_rate_rpm * 2 * pi / 60
    actuators_with_speed = [
        AzimuthThrusterWithSpeed(
            (33, 0), 0, (-max_rate_rad_sec, max_rate_rad_sec), (0, 300), dt
        ),
        AzimuthThrusterWithSpeed(
            (-33, 0), 0, (-max_rate_rad_sec, max_rate_rad_sec), (0, 300), dt
        )
    ]
    
    actuators = [
        AzimuthThruster(
            (33, 0), 0, (-180, 180), (0, 300), dt
        ),
        AzimuthThruster(
            (-33, 0), 0, (-180, 180), (0, 300), dt
        )
    ]

    path_to_params = os.path.join('nav_env', 'ships', 'blindheim_risk_2020.json')

    ship1 = Ship(
        states=States3(x=wpts[0][0], y=wpts[0][1], x_dot=0.5, y_dot=2.7, psi_deg=-20),
        physics=phy.ShipPhysics(path_to_params),
        actuators=actuators_with_speed,
        guidance=PathProgressionAndSpeedGuidance(
            wpts, 3
        ),
        controller=NMPCPathTracking(
            route=wpts,
            physics=SimpleShipPhysics(path_to_params),
            actuators=actuators_with_speed,
            weights={
                "kappa": 1*np.array([20, 5e4, 1]).T,
                # "Lambda": 0e-2*np.diag([0, 1]),
                # "Delta": 0e-2*np.diag([1e-2, 1])
                "Lambda": 1*np.diag([5e1, 5e-3, 5e1, 5e-3]), # a_rate, speed, a_rate, speed
                "Delta": 1*np.diag([2e2, 1e5, 2e2, 1e5])
            },
            horizon=40,
            dt=dt
        )
    )

    ship2 = Ship(
        states=States3(x=wpts[0][0], y=wpts[0][1], x_dot=0.5, y_dot=2.7, psi_deg=-20),
        physics=phy.ShipPhysics(path_to_params),
        guidance=LOSLookAhead(
            waypoints=wpts,
            radius_of_acceptance=200.,
            current_wpt_idx=1,
            kp=1e-4, # 7e-3
            desired_speed=3.
        ),
        controller=HeadingAndSpeedController(
            pid_gains_heading=(2.5e5, 0, 1e5),
            pid_gains_speed=(8e4, 1e4, 0),
            dt=dt,
            allocation=NonlinearControlAllocation(actuators=actuators)
        ),
        actuators=actuators,
        # guidance=LOSLookAhead(
        #     waypoints=wpts,
        #     radius_of_acceptance=200.,
        #     current_wpt_idx=1,
        #     kp=1e-8, # 7e-3
        #     desired_speed=3.
        # ),
        # controller=HeadingAndSpeedController(
        #     pid_gains_heading=(2.5e5, 0, 1e5),
        #     pid_gains_speed=(8e4, 1e4, 0),
        #     dt=dt
        # ),
        name="ship2"
    )

    env = NavigationEnvironment(
        own_ships=[ship1, ship2],
        dt=dt,
        wind_source=UniformWindSource(0, 0)
    )

    x_lim = (-100, 2*R+100)
    y_lim = (-R-100, R+100)
    tf = 500

    sim = Simulator(env)
    sim.run(tf, dt, record_results_dt=10)
    sim.replay(x_lim=x_lim, y_lim=y_lim, speed=20)
    



if __name__ == "__main__":
    # test_path()
    test_path_but_with_proper_simulator()