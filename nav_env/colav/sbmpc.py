from nav_env.ships.ship import Ship, MovingShip
from nav_env.wind.wind_vector import WindVector
from nav_env.water.water_vector import WaterVector
from nav_env.obstacles.obstacles import Obstacle
from nav_env.control.command import Command
from nav_env.utils.math_functions import wrap_angle_to_pmpi
import numpy as np, math
from dataclasses import dataclass, field


@dataclass
class SBMPCParams:
    """Parameters for the SB-MPC algorithm."""

    P_: float = 1.0  # weights the importance of time until the event of collision occurs
    Q_: float = 4.0  # exponent to satisfy colregs rule 16
    D_INIT_: float = 1000.0  # should be >= D_CLOSE   # distance to an obstacle to activate sbmpc [m]
    D_CLOSE_: float = 400.0  # distance for an nearby obstacle [m]
    D_SAFE_: float = 200.0  # distance of safety zone [m]
    K_COLL_: float = 0.5  # Weight for cost of collision --> C_i^k = K_COLL * |v_os - v_i^k|^2
    PHI_AH_: float = np.deg2rad(68.5)  # colregs angle - ahead [deg]
    PHI_OT_: float = np.deg2rad(68.5)  # colregs angle - overtaken [deg]
    PHI_HO_: float = np.deg2rad(22.5)  # colregs angle -  head on [deg]
    PHI_CR_: float = np.deg2rad(68.5)  # colregs angle -  crossing [deg]
    KAPPA_: float = 10.0  # Weight for cost of COLREGs compliance (Rules 14 & 15, if both are satisfied it implies 13 is also satisfied)
    K_P_: float = 2.5  # Weight for penalizing speed offset
    K_CHI_: float = 1.5  # Weight for penalizing heading offset
    K_DP_: float = 2.0  # Weight for penalizing changes in speed offset
    K_DCHI_SB_: float = 1.0  # Weight for penalizing changes in heading offset in StarBoard situation
    K_DCHI_P_: float = 1.4  # Weight for penalizing changes in heading offset in Port situation

    # K_DCHI is greater in port than in starboard in compliance with COLREGS rules 14, 15 and 17.

    P_ca_last_: float = 1.0  # last control change
    Chi_ca_last_: float = 0.0  # last course change

    Chi_ca_: np.array = field(
        default_factory=lambda: np.deg2rad(
            np.array([-30.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 30.0])
        )
    )  # control behaviors - course offset [deg]
    P_ca_: np.array = field(default_factory=lambda: np.array([1.0]))  # control behaviors - speed factor

    def to_dict(self):
        output = {
            "P_": self.P_,
            "Q_": self.Q_,
            "D_INIT_": self.D_INIT_,
            "D_CLOSE_": self.D_CLOSE_,
            "D_SAFE_": self.D_SAFE_,
            "K_COLL_": self.K_COLL_,
            "PHI_AH_": self.PHI_AH_,
            "PHI_OT_": self.PHI_OT_,
            "PHI_HO_": self.PHI_HO_,
            "PHI_CR_": self.PHI_CR_,
            "KAPPA_": self.KAPPA_,
            "K_P_": self.K_P_,
            "K_CHI_": self.K_CHI_,
            "K_DP_": self.K_DP_,
            "K_DCHI_SB_": self.K_DCHI_SB_,
            "K_DCHI_P_": self.K_DCHI_P_,
            "P_ca_last": self.P_ca_last_,
            "Chi_ca_last": self.Chi_ca_last_,
            "Chi_ca_": self.Chi_ca_,
            "P_ca_": self.P_ca_,
        }
        return output

    @classmethod
    def from_dict(cls, data: dict):
        output = SBMPCParams(
            P_=data["P_"],
            Q_=data["Q_"],
            D_INIT_=data["D_INIT_"],
            D_CLOSE_=data["D_CLOSE_"],
            D_SAFE_=data["D_SAFE_"],
            K_COLL_=data["K_COLL_"],
            PHI_AH_=data["PHI_AH_"],
            PHI_OT_=data["PHI_OT_"],
            PHI_HO_=data["PHI_HO_"],
            PHI_CR_=data["PHI_CR_"],
            KAPPA_=data["KAPPA_"],
            K_P_=data["K_P_"],
            K_CHI_=data["K_CHI_"],
            K_DP_=data["K_DP_"],
            K_DCHI_SB_=data["K_DCHI_SB_"],
            K_DCHI_P_=data["K_DCHI_P_"],
            P_ca_last_=data["P_ca_last_"],
            Chi_ca_last_=data["Chi_ca_last_"],
            Chi_ca_=data["Chi_ca_"],
            P_ca_=data["P_ca_"],
        )
        return output
    


class SBMPC:
    def __init__(self, tf:float=150.0, dt:float=5.0, config: SBMPCParams = None) -> None:

        """

        """


        # NB os_ship: copy of own ship initialized class
        self.T_ = tf  # 400                       # prediction horizon [s]
        self.DT_ = dt  # 0.1                          # time step [s]
        self.n_samp = int(self.T_ / self.DT_)  # number of samplings

        self.cost_ = np.inf

        self.ownship = ShipModel(self.T_, self.DT_)
        # print("OWN SHIP: ", self.ownship)

        if config:
            self._params = config
        else:
            self._params = SBMPCParams()

    def get_optimal_ctrl_offset(
        self,
        u_d: float,
        chi_d: float,
        os_state: np.ndarray,
        do_list: list[tuple[int, np.ndarray, np.ndarray, float, float]]
        # enc: senc.ENC,
    ) -> tuple[float, float]:
        """Calculates the optimal control offset for the own ship using the SB-MPC algorithm.

        Args:
            u_d (float): Nominal surge speed reference for the own ship.
            chi_d (float): Nominal course reference for the own ship.
            os_state (np.ndarray): Current state of the own ship.
            do_list (List[Tuple[int, np.ndarray, np.ndarray, float, float]]): List of tuples containing the dynamic obstacle info
            enc (senc.ENC): Electronic navigational chart.

        Returns:
            Tuple[float, float]: Optimal control offset to the own ship nominal LOS references, (speed factor, course offset).
        """
        cost = np.inf
        cost_i = 0
        colav_active = False
        d = np.zeros(2)

        if do_list is None:
            u_os_best = 1
            chi_os_best = 0
            self._params.P_ca_last_ = 1
            self._params.Chi_ca_last_ = 0
            return u_os_best, chi_os_best
        else:
            obstacles = []
            n_obst = len(do_list)
            for obs_state in do_list:
                obstacle = Obstacle(obs_state, self.T_, self.DT_)
                obstacles.append(obstacle)

        # check if obstacles are within init range
        for obs in obstacles:
            d[0] = obs.x_[0] - os_state[0]
            d[1] = obs.y_[0] - os_state[1]
            # print(*d)
            if np.linalg.norm(d) < self._params.D_INIT_:
                colav_active = True
        # print("COLAV ACTIVE ? ", colav_active, np.linalg.norm(d))
        if not colav_active:
            u_os_best = 1
            chi_os_best = 0
            self._params.P_ca_last_ = 1
            self._params.Chi_ca_last_ = 0
            return u_os_best, chi_os_best

        for i in range(len(self._params.Chi_ca_)):
            for j in range(len(self._params.P_ca_)):
                self.ownship.linear_pred(os_state, u_d * self._params.P_ca_[j], chi_d + self._params.Chi_ca_[i])

                cost_i = -1
                for k in range(n_obst):
                    cost_k = self.cost_func(self._params.P_ca_[j], self._params.Chi_ca_[i], obstacles[k])
                    if cost_k > cost_i:
                        cost_i = cost_k
                if cost_i < cost:
                    cost = cost_i
                    u_os_best = self._params.P_ca_[j]
                    chi_os_best = self._params.Chi_ca_[i]

        # if self._params.Chi_ca_last_ != chi_os_best or self._params.P_ca_last_ != u_os_best:
        #     print('best: ', u_os_best, chi_os_best, '\n')

        self._params.P_ca_last_ = u_os_best
        self._params.Chi_ca_last_ = chi_os_best

        return u_os_best, chi_os_best

    def cost_func(self, P_ca: float, Chi_ca: float, obstacle:Obstacle):
        obs_l = obstacle.l
        obs_w = obstacle.w
        os_l = self.ownship.l
        os_w = self.ownship.w

        d, v_o, v_s = np.zeros(2), np.zeros(2), np.zeros(2)
        self.combined_radius = os_l + obs_l
        d_safe = self._params.D_SAFE_
        d_close = self._params.D_CLOSE_
        H0, H1, H2 = 0, 0, 0
        cost = 0
        t = 0
        t0 = 0

        for i in range(self.n_samp):

            t += self.DT_

            d[0] = obstacle.x_[i] - self.ownship.x_[i]
            d[1] = obstacle.y_[i] - self.ownship.y_[i]
            dist = np.linalg.norm(d)

            R = 0
            C = 0
            mu = 0

            if dist < d_close:
                v_o[0] = obstacle.u_[i]
                v_o[1] = obstacle.v_[i]
                v_o = self.rot2d(obstacle.psi_, v_o) # --> speed of obstacle in world frame

                v_s[0] = self.ownship.u_[i]
                v_s[1] = self.ownship.v_[i]
                v_s = self.rot2d(self.ownship.psi_[i], v_s) # --> speed of ownship in world frame

                psi_o = wrap_angle_to_pmpi(obstacle.psi_)
                phi_before_wrap = math.atan2(d[1], d[0]) - self.ownship.psi_[i] - math.pi/2
                phi = wrap_angle_to_pmpi(math.atan2(d[1], d[0]) - self.ownship.psi_[i] - math.pi/2)  # Difference between os heading and direction towards obstacle
                psi_rel = wrap_angle_to_pmpi(psi_o - self.ownship.psi_[i]) # relative heading of obstacle w.r.t ownship's heading

                if phi < self._params.PHI_AH_: # If the ship's heading is oriented too much towards the obstacle (i.e. in the AHEAD SECTOR)
                    d_safe_i = d_safe + os_l / 2 # Then we increase the safe distance by ownship's length
                elif phi > self._params.PHI_OT_: # If the ship's heading is not oriented towards the obstacle (i.e. in the OVERTAKING SECTOR)
                    d_safe_i = 0.5 * d_safe + os_l / 2 #  reduce safe distance
                else:
                    d_safe_i = d_safe + os_w / 2

                phi_o = wrap_angle_to_pmpi(math.atan2(-d[1], -d[0]) - obstacle.psi_ + math.pi/2) 

                if phi_o < self._params.PHI_AH_:
                    d_safe_i = d_safe + obs_l / 2
                elif phi_o > self._params.PHI_OT_:
                    d_safe_i = 0.5 * d_safe + obs_l / 2
                else:
                    d_safe_i = d_safe + obs_w / 2


                
                if (np.dot(v_s, v_o)) > np.cos(np.deg2rad(self._params.PHI_OT_)) * np.linalg.norm(v_s) * np.linalg.norm(
                    v_o
                ) and np.linalg.norm(v_s) > np.linalg.norm(v_o):
                    d_safe_i = d_safe + os_l / 2 + obs_l / 2 # --> Increases safety distance

                if dist < d_safe_i:
                    R = (1 / (abs(t - t0) ** self._params.P_)) * (d_safe / dist) ** self._params.Q_
                    k_koll = self._params.K_COLL_ * os_l * obs_l
                    C = k_koll * np.linalg.norm(v_s - v_o) ** 2

                # Overtaken by obstacle
                # Those conditions checks if the ship is being OVERTAKEN by the obstacle
                # first condition checks if the angle between their speed is greater than the overtaking angle (PHI_OT)
                # second condition checks if the ship has greater speed than the obstacle
                OT = (np.dot(v_s, v_o)) > np.cos(np.deg2rad(self._params.PHI_OT_)) * np.linalg.norm(
                    v_s
                ) * np.linalg.norm(v_o) and np.linalg.norm(v_s) < np.linalg.norm(v_o)

                # Obstacle on starboard side
                SB = phi >= 0

                # Obstacle Head-on
                HO = (
                    np.linalg.norm(v_o) > 0.05
                    and (np.dot(v_s, v_o))
                    < -np.cos(np.deg2rad(self._params.PHI_HO_)) * np.linalg.norm(v_s) * np.linalg.norm(v_o)
                    and (np.dot(v_s, v_o)) > np.cos(np.deg2rad(self._params.PHI_AH_)) * np.linalg.norm(v_s)
                )

                # Crossing situation
                CR = (np.dot(v_s, v_o)) < np.cos(np.deg2rad(self._params.PHI_CR_)) * np.linalg.norm(
                    v_s
                ) * np.linalg.norm(v_o) and (SB and psi_rel < 0)

                mu = (SB and HO) or (CR and not OT)

            H0 = C * R + self._params.KAPPA_ * mu

            if H0 > H1:
                H1 = H0

        H2 = (
            self._params.K_P_ * (1 - P_ca)
            + self._params.K_CHI_ * Chi_ca**2
            + self.delta_P(P_ca)
            + self.delta_Chi(Chi_ca)
        )
        cost = H1 + H2

        return cost

    def delta_P(self, P_ca):
        return self._params.K_DP_ * abs(self._params.P_ca_last_ - P_ca)

    def delta_Chi(self, Chi_ca):
        d_chi = Chi_ca - self._params.Chi_ca_last_
        if d_chi > 0:
            return self._params.K_DCHI_SB_ * d_chi**2
        elif d_chi < 0:
            return self._params.K_DCHI_P_ * d_chi**2
        else:
            return 0

    def rot2d(self, yaw: float, vec: np.ndarray):
        R = np.array([[-np.sin(yaw), np.cos(yaw)], [np.cos(yaw), np.sin(yaw)]])
        return R @ vec
    
class Obstacle:
    def __init__(self, state: np.ndarray, T: np.double, dt: np.double):
        self.n_samp_ = int(T / dt)

        self.T_ = T
        self.dt_ = dt

        self.x_ = np.zeros(self.n_samp_)
        self.y_ = np.zeros(self.n_samp_)
        self.u_ = np.zeros(self.n_samp_)
        self.v_ = np.zeros(self.n_samp_)


        self.x_[0] = state[1][0]
        self.y_[0] = state[1][1]
        V_x = state[1][2]
        V_y = state[1][3]
        self.psi_ = np.arctan2(V_y, V_x) - math.pi/2  # chi

        self.l = state[3]
        self.w = state[4]

        self.r11_ = -np.sin(self.psi_) # --> Rotation matrix to bring uvr into world, using the desired heading
        self.r12_ = np.cos(self.psi_)
        self.r21_ = np.cos(self.psi_)
        self.r22_ = np.sin(self.psi_)

        # u = Vy * cos(psi) - Vx * sin(psi)
        # v = Vy * sin(psi) + Vx * cos(psi)

        self.u_[0] = self.r11_ * V_x + self.r12_ * V_y
        self.v_[0] = self.r21_ * V_x + self.r22_ * V_y

        self.calculate_trajectory()

    def calculate_trajectory(self):
        for i in range(1, self.n_samp_):
            self.x_[i] = self.x_[i - 1] + (self.r11_ * self.u_[i - 1] + self.r12_ * self.v_[i - 1]) * self.dt_
            self.y_[i] = self.y_[i - 1] + (self.r21_ * self.u_[i - 1] + self.r22_ * self.v_[i - 1]) * self.dt_
            self.u_[i] = self.u_[i - 1]
            self.v_[i] = self.v_[i - 1]
    
class ShipModel:
    def __init__(self, T: np.double, dt: np.double, length:float=25, width:float=80):
        self.n_samp_ = int(T / dt)

        self.T_ = T
        self.DT_ = dt

        self.x_ = np.zeros(self.n_samp_)
        self.y_ = np.zeros(self.n_samp_)
        self.psi_ = np.zeros(self.n_samp_)
        self.u_ = np.zeros(self.n_samp_)
        self.v_ = np.zeros(self.n_samp_)
        self.r_ = np.zeros(self.n_samp_)

        self.l = length
        self.w = width


    def linear_pred(self, state, u_d, psi_d):
        self.psi_[0] = wrap_angle_to_pmpi(psi_d)

        self.x_[0] = state[0]
        self.y_[0] = state[1]
        self.u_[0] = u_d
        self.v_[0] = state[4]
        self.r_[0] = 0

        # r11 = np.cos(psi_d) # --> Rotation matrix to bring uvr into world, using the desired heading
        # r12 = -np.sin(psi_d)
        # r21 = np.sin(psi_d)
        # r22 = np.cos(psi_d)

        r11 = -np.sin(psi_d) # --> Rotation matrix to bring uvr into world, using the desired heading
        r12 = np.cos(psi_d)
        r21 = np.cos(psi_d)
        r22 = np.sin(psi_d)

        for i in range(1, self.n_samp_):
            self.x_[i] = self.x_[i - 1] + self.DT_ * (r11 * self.u_[i - 1] + r12 * self.v_[i - 1]) # Output is in world frame
            self.y_[i] = self.y_[i - 1] + self.DT_ * (r21 * self.u_[i - 1] + r22 * self.v_[i - 1])
            self.psi_[i] = psi_d  # self.psi_[i-1] + self.DT_*self.r_[i-1]
            self.u_[i] = u_d  # self.u_[i-1] + self.DT_*(u_d-self.u_[i-1])
            self.v_[i] = 0
            self.r_[i] = 0  # math.atan2(np.sin(psi_d - self.psi_[i-1]), np.cos(psi_d - self.psi_[i-1]))


# class SBMPC:
#     def __init__(
#         self,
#         *args,
#         t_horizon:float=150.0,
#         dt:float=1.0,
#         config:SBMPCParams=SBMPCParams(),
#         **kwargs
#     ) -> None:
#         """
        
#         Scenario are parametrized with two features:
#             - Course angle commanded to the autopilot
#             - Propulsion command from nominal to full reverse
        
#         """
#         self.t_horizon = t_horizon
#         self.dt = dt
#         self.n_samples = int(self.t_horizon // self.dt)
#         self.params = config

#         # print(self.headings)
#         # print(self.speeds)

#     # def reset_cost_of_scenarios(self) -> None:
#     #     for scenario in self.scenarios:
#     #         scenario["cost"] = 0.0

#     def cost_fn(self, speed:float, heading:float, target_ship:MovingShip, *args, **kwargs) -> float:
#         D_SAFE = 300
#         D_CLOSE = 
#         min_dist = float('inf')
#         for n in range(self.horizon):
#             t = (n+1) * self.own_ship.dt
#             pose = target_ship.linear_prediction_from_current_state(t)
#             enveloppe = target_ship.enveloppe_fn_from_linear_prediction(t)
#             dist = enveloppe.distance(self.os_enveloppes[n])
#             if dist < min_dist:
#                 min_dist = dist
#             # print(f"DIST AT TIME {t}: ", dist)
#             # target
#         if min_dist > safe_dist:
#             return 0.0
#         else:
#             return np.exp(-min_dist)
#         # for n in range(self.horizon):
#         #     t = (n+1)*own_ship.dt
#         #     pose_os = own_ship.pose_fn_from_current_state_given_u_and_psi(t, speed, heading)
#         #     pose_tss = []
#         #     pose_i = target_ship.pose_fn_from_current_state(t)
#         #     pose_tss.append(pose_i)

                

#     def get(self, own_ship_in:Ship, target_ships:list[MovingShip], wind:WindVector=WindVector((0., 0.), vector=(0., 0.)), water:WaterVector=WaterVector((0., 0.), vector=(0., 0.))) -> tuple[float, float]:
#         """
#         Returns a tuple containing best speed and heading commands according to SB-MPC
        
#         """
#         self.own_ship = deepcopy(own_ship_in)
#         target_ships = [deepcopy(ts) for ts in target_ships]
#         desired_states = own_ship._gnc.goal()
#         u_d = desired_states.x_dot
#         psi_d = desired_states.psi_deg

#         ### Simulate scenarios over horizon with TSs as linear obstacles, evaluate cost, keep the best
#         # self.reset_cost_of_scenarios()
#         cost = float('inf')
#         for i in range(len(self.headings)):
#             for j in range(len(self.speeds)):
                
#                 # Compute ship's next state given u_d, psi_d
#                 self.os_states = []
#                 self.os_enveloppes = []
#                 for n in range(self.horizon):
#                     t = (n+1)*self.own_ship.dt
#                     os_state:States3 = self.own_ship.linear_prediction_from_current_state_given_u_and_psi(t, u_d, psi_d)
#                     os_enveloppe:States3 = self.own_ship.enveloppe_fn_from_linear_prediction_given_u_and_psi(t, u_d, psi_d)
#                     self.os_states.append(os_state)
#                     self.os_enveloppes.append(os_enveloppe)

#                 # Compute cost
#                 cost_i = -1
#                 for k in range(len(target_ships)):
#                     cost_k = self.cost_fn(self.speeds[j] * u_d, self.headings[i] + psi_d, target_ships[k])
#                     if cost_k > cost_i: # Once cost gets greater than best cost, we stop evaluating the scenario for improved efficiency
#                         cost_i = cost_k
#                 if cost_i <= cost: # If total cost of current scenario is better than previous, keep it
#                     cost = cost_i
#                     best_speed_change, best_heading_change = self.speeds[j], self.headings[i]

#         if cost == 0.0:
#             best_speed_change, best_heading_change = 1, 0
        
#         self._last_speed, self._last_heading = best_speed_change, best_heading_change 
#         return best_speed_change, best_heading_change

    # @property
    # def n_scenarios(self) -> int:
    #     return len(self.scenarios)

if __name__ == "__main__":
    from nav_env.ships.ship import Ship, SimpleShip
    from nav_env.ships.sailing_ship import SailingShip
    from nav_env.ships.states import States3
    from nav_env.control.LOS import LOSLookAhead
    from nav_env.control.PID import HeadingAndSpeedController
    import matplotlib.pyplot as plt
    from nav_env.environment.environment import NavigationEnvironment

    dt = 1.
    wpts = [
        (100., 300.),
        (1500., 1800.),
        (3000., 3000.)
    ]

    # At first let's keep sbmpc outside of the framework
    # Once it works, we can integrate it 

    sbmpc = SBMPC()

    own_ship = Ship(
        states=States3(0, 50),
        guidance=LOSLookAhead(
            waypoints=wpts,
            radius_of_acceptance=100.,
            current_wpt_idx=1,
            kp=3e-4, # 7e-3
            desired_speed=3.
        ),
        controller=HeadingAndSpeedController(
            pid_gains_heading=(2.5e5, 0, 1e7),
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

    ts1 = SailingShip(length=75, width=25, initial_state=States3(100, 1500, x_dot=1.7, y_dot=-1.8))


    

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

        print(f"{own_ship._gnc.goal().psi_deg:.2f}\t {own_ship._gnc._colav_heading_offset_rad*180/math.pi:.2f}")
        # print(f"")
        colav_speed_factor, colav_heading_offset = sbmpc.get_optimal_ctrl_offset(
            own_ship._gnc.goal().x_dot,
            own_ship._gnc.goal().psi_rad,
            own_ship.states_in_ship_frame.to_numpy(),
            do_list=[(i, np.array([ts.states.x, ts.states.y, ts.states_in_ship_frame.x_dot, ts.states_in_ship_frame.y_dot]), None, ts.length, ts.width) for i, ts in enumerate([ts1])]
        )
        own_ship._gnc._colav_u_factor = colav_speed_factor
        own_ship._gnc._colav_heading_offset_rad = colav_heading_offset

        # print(own_ship._gnc.goal())
        # print("SBMPC.GET: ", )
        own_ship.enveloppe_fn_from_linear_prediction(100).plot(ax=ax)
        v = np.linalg.norm(own_ship.states.xy_dot)
        # print(v)
        if t%10 > 0:
            x.append(own_ship.states.x)
            y.append(own_ship.states.y)
        ax.plot(x, y, '--r')
        env.plot(lim, ax=ax)
        plt.pause(1e-9)

    plt.pause()



