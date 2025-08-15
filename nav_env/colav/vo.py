from nav_env.colav.colav import COLAVBase
from nav_env.ships.states import States3
from nav_env.ships.moving_ship import MovingShip
from nav_env.utils.math_functions import wrap_angle_to_pmpi
from math import sin, asin, pi, atan2, acos
import warnings, numpy as np, matplotlib.pyplot as plt
from typing import Literal
from nav_env.estimation.filter import LowPass
from shapely import LineString
from nav_env.obstacles.obstacles import Obstacle

DEFAULT_LOW_PASS_FILTER_PARAMS = {'cutoff':1, 'sampling_frequency':100, 'order':0}

class VelocityObstacle(COLAVBase):
    def __init__(
            self,
            distance_threshold:float,
            distance_margin:float,          # Margin w.r.t to the target ship's envelope
            *args,
            turn_heuristic:Literal['closest', 'behind', 'avoid shore']='closest',
            low_pass_filter_params:dict=None,
            d_shore:float=1000,
            t_enable:float=250, # [sec] If collision occurs in <= t_enable, add to velocity obstacle
            t_disable:float=0,  # [sec]
            **kwargs
    ):
        self.active = False 
        self.turn_heuristic = turn_heuristic                                # Whether the VO algorithm is currently active or not
        low_pass_filter_params = low_pass_filter_params or DEFAULT_LOW_PASS_FILTER_PARAMS
        self.filter = LowPass(**low_pass_filter_params)
        self.collision_coordinate = []
        self.d_shore = d_shore
        self.t_enable = t_enable
        self.t_disable = t_disable
        super().__init__(distance_threshold, *args, distance_margin=distance_margin, **kwargs)                           

    def reset(self) -> None:
        pass

    def merge_intervals(self, intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Merge overlapping or contiguous intervals on [-π, π) range."""        
        # Split wrapped intervals
        split_intervals = []
        for start, end in intervals:
            if start <= end:
                split_intervals.append((start, end))
            else:
                # Wraps around π/-π, split into two
                split_intervals.append((start, pi))
                split_intervals.append((-pi, end))
        
        # Sort and merge
        split_intervals.sort()
        merged = []
        for start, end in split_intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        
        return [(a*180/pi, b*180/pi) for a, b in merged]
    
    def get_closest_safe_turning_angle_degree(self, xy:tuple, desired_heading_deg:float, current_heading_deg:float, intervals: list[tuple[float, float]], active_target_ships_heading_and_dist:list[tuple[float, float]]=[], shore:list[Obstacle]=[]) -> float:
        if self.turn_heuristic == 'closest':
            # heading_mean_deg = 0.5*(desired_heading_deg + current_heading_deg)
            for start, end in intervals:
                if start <= desired_heading_deg <= end:
                    if abs(start-current_heading_deg) < abs(end-current_heading_deg):
                        return start-desired_heading_deg
                    else:
                        return end-desired_heading_deg
        elif self.turn_heuristic == 'avoid shore':
            # heading_mean_deg = 0.5*(desired_heading_deg + current_heading_deg)
            
            for start, end in intervals:
                if start <= desired_heading_deg <= end:
                    # angle_margin = 15
                    line_end = LineString([
                                            xy,
                                            (xy[0]-self.d_shore*np.sin((end)*np.pi/180), xy[1]+self.d_shore*np.cos(end*np.pi/180))
                                        ])
                    line_start = LineString([
                                            xy,
                                            (xy[0]-self.d_shore*np.sin((start)*np.pi/180), xy[1]+self.d_shore*np.cos(start*np.pi/180))
                                        ])
                    start_intersect = False
                    end_intersect = False
                    for obs in shore:
                        if obs.intersects(line_start):
                            # print("intersection: ", obs.intersection(line_start).get_xy_as_list()) 
                            start_intersect = True
                        if obs.intersects(line_end):
                            # print("intersection: ", obs.intersection(line_end).get_xy_as_list()) 
                            end_intersect = True
                    # if start_intersect or end_intersect:
                    #     print("start: ", start_intersect, f"({start:.1f})", "end: ", end_intersect, f"({end:.1f})")

                    if not start_intersect and not end_intersect: # No intersection -> pick the closest to current heading
                        if abs(start-current_heading_deg) < abs(end-current_heading_deg): # start is the closest
                            return start-desired_heading_deg
                        else: # end is the closest
                            return end-desired_heading_deg
                    elif not start_intersect: # end intersect but not start
                        return start-desired_heading_deg
                    elif not end_intersect: # start intersect but not end
                        return end-desired_heading_deg
                    else: # Compromise
                         if abs(start-current_heading_deg) < abs(end-current_heading_deg):
                             return (3*start+end)/4-desired_heading_deg
                         else:
                             return (3*end+start)/4-desired_heading_deg
                         
        elif self.turn_heuristic == 'behind' and len(active_target_ships_heading_and_dist)>0:
            raise NotImplementedError(f"this heuristic is probably inconsistent in this form. Depending on the OS's relative pose w.r.t TS, it does not make any sense to pass behind.")
            # print(active_target_ships_heading_and_dist)
            heading, dist = zip(*active_target_ships_heading_and_dist)
            idx = np.argmin(dist)
            for start, end in intervals:
                if start <= heading_deg <= end:
                    if abs(heading[idx]-heading_deg) < 90:
                        return start-heading_deg
                    else:
                        return end-heading_deg
        return 0.0

    def __get__(self, state:States3, commanded_state:States3, target_ships:list[MovingShip], *args, degree:bool=True, shore:list=[], **kwargs) -> States3:
        """
        degree:bool control if input is given in degrees (True) or not (False)
        """
        self.active = False 
        u_d, psi_d = commanded_state.x_dot, commanded_state.psi_deg #(commanded_state.x_dot*0.2 + state.x_dot*0.8), (commanded_state.psi_deg*0.2 + state.psi_deg*0.8)
        # angle_to_degrees = 1 if degree else 180*psi_d/pi
        x_os, y_os = state.xy
        desired_velocity_os_world = np.array(States3(psi_deg=psi_d, u=u_d).xy_dot).reshape((2, 1))
        v = np.linalg.norm(desired_velocity_os_world)

        # list of tuple defining the critical course angle values (theta_min, theta_max)
        critical_ranges = []
        active_target_ships_heading_and_dist = []

        self.collision_coordinate = []
        for i, ts in enumerate(target_ships): 
            # print(f"VELOCITY OBSTACLE {i}")
            # Build circular enveloppe around ts
            l, w = ts.length, ts.width
            r0 = ((l/2)**2+(w/2)**2)**0.5           # Circular envelope around target ship
            r = r0 + self.distance_margin   # Margin around circular envelope
            # print(r)
            
            # Distance between ts and os
            dxy = np.array([ts.states.x-x_os, ts.states.y-y_os]).reshape((2, 1))
            d = np.linalg.norm(dxy)

            # Check if ts is close enough to care about
            if d <= self.distance_threshold:
                self.active = True
                # print("COLAV ACTIVE")

                # Check if ts is not already colliding with us

                # Compute the critical angle. If theta is less than this angle, our current speed will lead to a collision
                theta_cr = asin(np.clip(r/d, -1, 1))
                
                # Compute relative velocity of os w.r.t ts
                velocity_ts_world = np.array(ts.states.xy_dot).reshape((2, 1))
                angle_of_ts_velocity_world = atan2(-ts.states.x_dot, ts.states.y_dot)
                velocity_rel_world = desired_velocity_os_world - velocity_ts_world
                v_rel_world = np.linalg.norm(velocity_rel_world)

                # TCPA
                t_star = (dxy.T @ velocity_rel_world)[0, 0] / v_rel_world**2
                # print("t_star: ", t_star)
                # d_star = np.linalg.norm(dxy - velocity_rel_world * t_star)
                # print("d_star: ", d_star)
                self.collision_coordinate.append((x_os + desired_velocity_os_world[0] * t_star, y_os + desired_velocity_os_world[1] * t_star))
                if self.t_disable <= t_star <= self.t_enable: # If collision occurs backward in time, we just skip it
                    active_target_ships_heading_and_dist.append((ts.states.psi_deg, d))
                    # angle of direction vector in counter-clockwise world frame
                    relative_vector_angle_in_world = atan2(-dxy[0, 0], dxy[1, 0])

                    # range of VO in world
                    theta_1 = -theta_cr
                    theta_2 = theta_cr

                    # Betas
                    beta_0 = relative_vector_angle_in_world
                    beta_1 = beta_0 - theta_1
                    beta_2 = beta_0 - theta_2

                    # Ds
                    Omega = -angle_of_ts_velocity_world
                    D_1 = pi - beta_1 - Omega
                    D_2 = pi - beta_2 - Omega

                    # Es
                    vb = np.linalg.norm(velocity_ts_world)
                    E_1 = asin(np.clip(vb*sin(D_1)/v, -1, 1))
                    E_2 = asin(np.clip(vb*sin(D_2)/v, -1, 1))

                    # phis
                    phi_1 = wrap_angle_to_pmpi(beta_1-E_1)
                    phi_2 = wrap_angle_to_pmpi(beta_2-E_2)
                    # phi_min, phi_max = min(phi_1, phi_2), max(phi_1, phi_2)

                    # print(f"(i={i})" , f"\tt_star: {t_star:.1f}", f"d(t_star): {d_star:.1f}", f"theta cr: {180*theta_cr/pi:.1f}", f"theta: {180*acos(velocity_rel_world.T@dxy/(v_rel_world*d))/pi:.1f}", f"phi: [{180*phi_2/pi:.1f}, {180*phi_1/pi:.1f}]")

                    critical_ranges.append((phi_2, phi_1))                        
                    # _, ax = plt.subplots()
                    # ax.quiver(0, 0, *velocity_ts_world.tolist(), angles='xy', scale_units='xy', scale=1, label="vb", color='blue')
                    # ax.quiver(0, 0, *desired_velocity_os_world.tolist(), angles='xy', scale_units='xy', scale=1, label="v")
                    # ax.quiver(*velocity_ts_world.tolist(), *velocity_rel_world.tolist(), angles='xy', scale_units='xy', scale=1, label="dv", color='red')
                    # ax.quiver(0, 0, *dxy.tolist(), angles='xy', scale_units='xy', scale=10, label="dp", color='green')
                    # ax.quiver(0, 0, *velocity_rel_world.tolist(), angles='xy', scale_units='xy', scale=1, label="dv", color='red')
                    # ax.set_xlim([-5, 5])
                    # ax.set_ylim([-5, 5])
                    # ax.legend()
                    # ax.set_aspect('equal')
                    # plt.show()
                # else:
                #     # critical_ranges = [(-pi/2, pi/2)]
                #     warnings.warn(f"Entering safe zone of target ship {i} (d={d:.1f}<{r:.1f}=r)")
                #     continue
            else:
                continue
        # print(self.merge_intervals(critical_ranges))
        # print()
        # print("critical ranges: ", [(critical_range[0]*180/pi, critical_range[1]*180/pi) for critical_range in critical_ranges])
        unfiltered = self.get_closest_safe_turning_angle_degree((x_os, y_os), psi_d, state.psi_deg, self.merge_intervals(critical_ranges), active_target_ships_heading_and_dist=active_target_ships_heading_and_dist, shore=shore)
        delta_psi_deg = self.filter(unfiltered)
        # print("unfiltered: ", unfiltered, "delta_psi_deg: ", delta_psi_deg)
        # print("desired heading in colav: ", psi_d + delta_psi_deg)
        # print("delta psi deg: ", delta_psi_deg)
        return States3(x=commanded_state.x, y=commanded_state.y, x_dot=u_d, psi_deg=psi_d + delta_psi_deg)
    

def test_vo() -> None:
    from nav_env.ships.moving_ship import MovingShip
    from nav_env.ships.states import States3
    import matplotlib.pyplot as plt
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.simulation.simulator import Simulator
    from nav_env.obstacles.obstacles import Circle

    psi_deg_des = -130
    u_des = 3
    state_des = States3(psi_deg=psi_deg_des, u=u_des)
    own_ship = MovingShip(states=States3(-20, 80, psi_deg_des, u=u_des)) # -16.958, 44.97
    ship2 = MovingShip(states=States3(40, 0, 10, u=2), domain=Circle(0, 0, 20.77))
    ship3 = MovingShip(states=States3(0, 40, 90, u=1), domain=Circle(0, 0, 20.77))
    
    VO = VelocityObstacle(
        distance_threshold=50, 
        distance_margin=10,
        own_ship=own_ship
    )

    dt = 0.1

    ax = own_ship(0).plot()
    ship2(0).plot(ax=ax)
    ship2.domain.plot(ax=ax)
    ship3(0).plot(ax=ax)
    ship3.domain.plot(ax=ax)

    ax.set_xlim([-30, 80])
    ax.set_ylim([-30, 80])
    ax.set_aspect('equal')
    plt.show()

    # Environment
    env = Env(
        own_ships=[own_ship],
        target_ships=[ship2, ship3],
        dt=dt
    )


    lim = ((-50, -50), (100, 100))
    ax = env.plot(lim)
    plt.show(block=False)
    x, y = [], []

    
    tf = 60
    for t in np.linspace(0, tf, int(tf//dt)):
        ax.cla()
        

        colav_heading_offset = VO.get(own_ship.states, state_des, [ship2, ship3]).psi_deg
        # colav_speed_factor, colav_heading_offset = sbmpc.get_optimal_ctrl_offset(
        #     own_ship._gnc.goal().x_dot,
        #     own_ship._gnc.goal().psi_rad,
        #     own_ship.states_in_ship_frame.to_numpy(),
        #     do_list=[(i, np.array([ts.states.x, ts.states.y, ts.states_in_ship_frame.x_dot, ts.states_in_ship_frame.y_dot]), None, ts.length, ts.width) for i, ts in enumerate([ts1])]
        # )
        
        s = own_ship.states
        own_ship = MovingShip(states=States3(s.x, s.y, psi_deg_des+colav_heading_offset, u=u_des))
        env.own_ships[0] = own_ship
        VO.own_ship = own_ship
        # print(psi_deg_des + colav_heading_offset, env.own_ships[0].states.psi_deg, VO.own_ship.states.psi_deg)
        # VO.own_ship.states = own_ship.states
        # own_ship._gnc._colav_heading_offset_rad = colav_heading_offset
        env.step()
        # print(own_ship._gnc.goal())
        # print("SBMPC.GET: ", )
        # own_ship.enveloppe_fn_from_linear_prediction(100).plot(ax=ax)
        # v = np.linalg.norm(own_ship.states.xy_dot)
        # print(v)
        # if t%10 > 0:
        #     x.append(own_ship.states.x)
        #     y.append(own_ship.states.y)
        # ax.plot(x, y, '--r')
        env.plot(lim, ax=ax)
        plt.pause(1e-9)

    plt.waitforbuttonpress()

    # sim = Simulator(env=env)
    # sim.run(tf=30, dt=0.5)
    # sim.replay([-10, 80], [-10, 80], speed=2)

def test_vo_within_framework() -> None:
    from nav_env.ships.ship import Ship, SimpleShip
    from nav_env.ships.sailing_ship import SailingShip
    from nav_env.ships.states import States3
    from nav_env.control.LOS import LOSLookAhead
    from nav_env.control.PID import HeadingAndSpeedController
    import matplotlib.pyplot as plt
    from nav_env.environment.environment import NavigationEnvironment
    from nav_env.obstacles.obstacles import Circle

    dt = 1.
    wpts = [
        (10., 30.),
        (500., 800.),
        (1000., 1600.)
    ]

    # At first let's keep sbmpc outside of the framework
    # Once it works, we can integrate it 

    own_ship = Ship(
        states=States3(20, 100, psi_deg=-30),
        guidance=LOSLookAhead(
            waypoints=wpts,
            radius_of_acceptance=100.,
            current_wpt_idx=1,
            kp=1e-2, # 7e-3
            desired_speed=3.,
            colav=VelocityObstacle(
                distance_threshold=500, 
                distance_margin=50
            )
        ),
        controller=HeadingAndSpeedController(
            pid_gains_heading=(1e5, 0, 2e8), # (2.5e5, 0, 1e7),
            pid_gains_speed=(8e4, 1e4, 0),
            dt=dt
        ),
        # guidance=LOSLookAhead(
        #     waypoints=wpts,
        #     radius_of_acceptance=100.,
        #     current_wpt_idx=1,
        #     kp=3e-4, # 7e-3
        #     desired_speed=3.
        # ),
        # controller=HeadingAndSpeedController(
        #     pid_gains_heading=(5e5, 0, 5e6),
        #     pid_gains_speed=(8e4, 1e4, 0),
        #     dt=dt
        # ),
        name="OS"
    )

    ts1 = SailingShip(length=75, width=25, initial_state=States3(100, 800, x_dot=1., y_dot=-1.2), domain=Circle(0, 0, radius=(75**2+25**2)**0.5+50))

    
    

    env = NavigationEnvironment(
        own_ships=[own_ship],
        target_ships=[ts1],
        dt=dt
    )



    #### IL FAUT QUE JE CHECK CE QUI SE PASSE AVEC MON CONTROLEUR LOS --> IL Y A UN PROBLEME D'ANGLE JE PENSE QUELQUE CHOSE QUI DEPASSE 2PI OU QQUE CHOSE SIMILAIRE



    lim = ((-50, -50), (2000, 2000))
    ax = env.plot(lim)
    plt.show(block=False)
    x, y = [], []

    tf = 5000
    for t in np.linspace(0, tf, int(tf//dt)):
        ax.cla()
        for wpt in wpts:
            ax.scatter(*wpt, c='black')
        ax.scatter(*own_ship._gnc._guidance.current_waypoint, c='red')
        ax.set_title(f"{t:.2f}")
        env.step()
        print("des state: ", env._own_ships[0]._gnc._desired_state, "state: ", env._own_ships[0].states_in_ship_frame.x_dot)
        
        # print(own_ship._gnc.goal())
        # print("SBMPC.GET: ", )
        # own_ship.enveloppe_fn_from_linear_prediction(100).plot(ax=ax)
        # v = np.linalg.norm(own_ship.states.xy_dot)
        # print(v)
        # if t%10 > 0:
        #     x.append(own_ship.states.x)
        #     y.append(own_ship.states.y)
        # ax.plot(x, y, '--r')

        for collision_coord in own_ship._gnc._guidance.colav.collision_coordinate:
            ax.scatter(*collision_coord, c='black')
        # ax.scatter(own_ship._gnc._)
        env.plot(lim, ax=ax, target_ships_physics={'enveloppe': 1, 'domain': 1})
        plt.pause(1e-9)

    plt.pause()


if __name__ == "__main__":
    test_vo_within_framework()
    # test_vo()