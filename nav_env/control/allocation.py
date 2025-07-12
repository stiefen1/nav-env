from nav_env.control.command import GeneralizedForces, Command
from nav_env.actuators.collection import ActuatorCollection
from typing import Union
from abc import ABC, abstractmethod
import casadi as cd, numpy as np, warnings
from copy import deepcopy



"""

MAYBE IT WOULD BE SMART TO LOOK AT THIS SURVEY FROM T.A. JOHANSEN (3rd most cited): https://www.sciencedirect.com/science/article/pii/S0005109813000368

"""



class ControlAllocationBase(ABC):
    """
    Input is a desired force, output is a list of actuator commands
    
    """
    def __init__(self, actuators:ActuatorCollection, *args, **kwargs):
        self.actuators = actuators
        

    # Setup the control allocation problem in any ways
    @abstractmethod
    def get(self, force:GeneralizedForces, *args, **kwargs) -> Union[list[Command], GeneralizedForces]:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
class ControlAllocation(ControlAllocationBase):
    def __init__(self, actuators:Union[ActuatorCollection, list]=ActuatorCollection.empty(), *args, **kwargs):
        actuators = actuators if isinstance(actuators, ActuatorCollection) else ActuatorCollection(actuators)
        super().__init__(actuators=actuators, *args, **kwargs)
    
    # Setup the control allocation problem in any ways
    def get(self, force:Union[GeneralizedForces, list[Command]], *args, **kwargs) -> Union[list[Command], GeneralizedForces]:       
        return force

    
    def reset(self) -> None:
        pass
    
    @staticmethod
    def empty() -> "ControlAllocation":
        return ControlAllocation(ActuatorCollection.empty())
    

class NonlinearControlAllocation(ControlAllocation):
    def __init__(self, actuators:Union[ActuatorCollection, list], *args, Q:np.ndarray=None, **kwargs):
        super().__init__(actuators=actuators, *args, **kwargs)
        self.Q = Q or np.diag(6 * [1.])
        self.reset()

    def reset(self) -> None:
        self._init_nlp()

    def _init_nlp(self) -> None:
        self.U_sym = cd.SX.sym("U", self.actuators.nu)
        self.LBU = self.actuators.u_min
        self.UBU = self.actuators.u_max
        self.cost_fn = lambda f, f_des, Q=self.Q: cd.mtimes(cd.mtimes(cd.transpose(f-f_des), Q), (f-f_des))
        
    def _solve_nlp(self, options:dict={'ipopt.print_level':0, 'print_time':False}) -> None:
        nlp = {
            "x": self.U_sym,
            "f": self.total_cost
        }

        solver_in = {
            "x0": self.actuators.u_mean,
            "lbx": self.LBU,
            "ubx": self.UBU
        }   

        self.solver = cd.nlpsol("allocation_solver", "ipopt", nlp, options)
        solver_out = self.solver(**solver_in)
        arr = solver_out['x'].full().flatten()
        self.Uopt = self.actuators.get_formated_commands(arr.reshape((self.actuators.nu,)).tolist())

    def get(self, force:GeneralizedForces, *args, **kwargs) -> list[Command]:
        """
        min (f(u)-f_des)^T@Q@(f(u)-f_des)
         u

        s.t.    f(u) == actuators.dynamics(u)
                u \in [u_min, u_max]
        """
        copy_of_actuators = deepcopy(self.actuators)
        total_force = copy_of_actuators.dynamics(self.U_sym, use_casadi=True, do_clip=False).to_casadi()
        self.total_cost = self.cost_fn(total_force, force.to_numpy())
        self._solve_nlp()
        return self.Uopt   
    
class PowerMinimizerControlAllocation(ControlAllocation):
    """
    min s'Qs + (u-u_pref)'W(u-u_pref)   # trade-off between small slack variable and energy cons.
    s.t.    tau_des - h(u, x, t) = s            # Match generated force to tau_des
            u \in U                             # Satisfy actuator's constraints
            u = u_prev + du                     # 
            du \in C                            # Satisfy rate constraints

    u_pref: Preferred value for u, i.e. value where energy consumption is minimal in this context
    s:      Slack variable
    u_prev: Previous value of u
    du:     Change of u
    C:      Set of feasible changes
    """
    def __init__(self, actuators:Union[ActuatorCollection, list], *args, Q:np.ndarray=None, W:np.ndarray=None, u_prev:tuple=None, **kwargs):
        super().__init__(actuators=actuators, *args, **kwargs)
        self.Q = np.eye(6)*1e-3 if Q is None else Q # If problem becomes sometimes infeasible, try decrease this weight matrix 
        self.n_forces = self.Q.shape[0]
        
        # Initialize previous commanded value
        if u_prev is not None:
            u_prev = np.array(u_prev)
        else:
            u_prev = np.array(self.actuators.u_mean)
        self.u_prev = u_prev.reshape((self.actuators.nu, 1))

        # Initialize weights and preferred values of u
        self.W, self.u_pref = self.actuators.get_weight_and_u_pref_for_power_minimimzation(W=W)

        # Cost function
        self.cost_fn = lambda u, s, u_pref=self.u_pref, Q=self.Q, W=self.W: cd.mtimes(cd.mtimes(cd.transpose(s), Q), s) + cd.mtimes(cd.mtimes(cd.transpose(u-u_pref), W), (u-u_pref))
        
        # Reset --> Init NLP
        self.reset()

    def reset(self) -> None:
        self._init_nlp()

    def _init_nlp(self) -> None:
        self.U_sym = cd.SX.sym("U", self.actuators.nu)
        self.S_sym = cd.SX.sym("S", self.n_forces) # Slack variables
        self.LBU = self.actuators.u_min
        self.UBU = self.actuators.u_max
        self.LBS = self.n_forces * [-cd.inf]
        self.UBS = self.n_forces * [cd.inf]
        self.LBG = cd.vertcat(
            self.n_forces * [0],
            np.array(self.actuators.u_rate_min) * self.actuators.dt
        )
        self.UBG = cd.vertcat(
            self.n_forces * [0],
            np.array(self.actuators.u_rate_max) * self.actuators.dt
        )

        print("u_min: ", np.array(self.actuators.u_rate_min) * self.actuators.dt)
        print("u_max: ", np.array(self.actuators.u_rate_max) * self.actuators.dt)
        self.s0 = tuple(self.n_forces * [0.0])
        
    def _solve_nlp(self, options:dict={'ipopt.print_level':0, 'print_time':False}) -> None:
        nlp = {
            "x": cd.vertcat(self.U_sym, self.S_sym),
            "f": self.total_cost,
            "g": self.G_sym
        }

        solver_in = {
            "x0": cd.vertcat(self.u_prev, self.s0),
            "lbx": cd.vertcat(self.LBU, self.LBS),
            "ubx": cd.vertcat(self.UBU, self.UBS),
            "lbg": self.LBG,
            "ubg": self.UBG
        }   

        self.solver = cd.nlpsol("allocation_solver", "ipopt", nlp, options)
        solver_out = self.solver(**solver_in)
        stats = self.solver.stats()
        if stats['return_status'] != 'Solve_Succeeded':
            warnings.warn(f"Optimization did not succeed: {stats['return_status']}")
        arr = solver_out['x'].full().flatten()
        uopt = arr[0:self.actuators.nu]
        sopt = arr[self.actuators.nu::]
        self.u_prev = uopt
        
        self.Uopt = self.actuators.get_formated_commands(uopt.reshape((self.actuators.nu,)).tolist())
        

    def get(self, force:GeneralizedForces, *args, **kwargs) -> list[Command]:
        copy_of_actuators = deepcopy(self.actuators)
        total_force = copy_of_actuators.dynamics(self.U_sym, use_casadi=True, do_clip=False).to_casadi()
        self.G_sym = cd.vertcat(
            force.to_numpy() - total_force - self.S_sym,
            self.U_sym - self.u_prev # Satisfy rate constraints
        )
        self.total_cost = self.cost_fn(self.U_sym, self.S_sym)
        self._solve_nlp()
        return self.Uopt  


def test() -> None:
    from nav_env.ships.ship import Ship
    from nav_env.ships.states import States3
    from nav_env.control.PID import HeadingAndSpeedController
    from nav_env.control.LOS import LOSLookAhead
    from nav_env.actuators.actuators import AzimuthThruster
    from nav_env.actuators.collection import ActuatorCollection
    import matplotlib.pyplot as plt
    from nav_env.environment.environment import NavigationEnvironment
    import numpy as np
    from nav_env.sensors.sensors import IMU

    wpts = [
        (0, 0),
        (200, 400),
        (480, 560),
        (900, 600),
        (1250, 950),
        (1500, 1500)
    ]

    dt = 1

    fig = plt.figure()
    ax = fig.add_subplot()
    for wpt in wpts:
        ax.scatter(*wpt, c='black')
    plt.show()

    actuators = ActuatorCollection([
        AzimuthThruster(
            (33, 0), 0, (-180, 180), (0, 300), dt, alpha_rate_max=30, v_rate_max=10
        ),
        AzimuthThruster(
            (-33, 0), 0, (-180, 180), (0, 300), dt, alpha_rate_max=30, v_rate_max=10
        )
            ])

    print("u_at_min_power: ", actuators.u_at_min_power())

    
    name="os"

    ship = Ship(
        states=States3(0, 50, x_dot=3, y_dot=3, psi_deg=-45),
        guidance=LOSLookAhead(
            waypoints=wpts,
            radius_of_acceptance=100.,
            current_wpt_idx=1,
            kp=3e-4, # 7e-3
            desired_speed=4.
        ),
        controller=HeadingAndSpeedController(
            pid_gains_heading=(5e5, 0, 5e6),
            pid_gains_speed=(5e5, 4e2, 0), # (1e5, 0, 0),
            dt=dt,
            # allocation=NonlinearControlAllocation(actuators=actuators)
            allocation=PowerMinimizerControlAllocation(actuators)#, W=np.eye(4)*1e-3)
        ),
        actuators=actuators,
        name=name,
        sensors=[IMU(src='states.pose'), IMU(src='states.uvr')]
    )
    
    env = NavigationEnvironment(
        own_ships=[ship],
        dt=dt
    )

    

    # ca = NonlinearControlAllocation(actuators=actuators)
    


    lim = ((-20, -20), (1800, 1800))
    ax = env.plot(lim)
    plt.show(block=False)
    x, y = [], []

    tf = 500
    for t in np.linspace(0, tf, int(tf//dt)):
        ax.cla()
        for wpt in wpts:
            ax.scatter(*wpt, c='black')
        ax.scatter(*ship._gnc._guidance.current_waypoint, c='red')
        
        ax.set_title(f"{t:.2f}")
        env.step()
        # commands = ca.get(ship._gnc._controller.last_commanded_force)
        # print("CONTROL ALLOCATION: ", commands)
        # print("Residual: ", ship._gnc._controller.last_commanded_force-actuators.dynamics(commands))
        # print(ship._gnc._controller.last_commanded_force)
        # v = np.linalg.norm(ship.states.xy_dot)
        v = ship.states.u
        print(v)
        if t%10 > 0:
            x.append(ship.states.x)
            y.append(ship.states.y)
        ax.plot(x, y, '--r')
        env.plot(lim, ax=ax)
        plt.pause(1e-9)

    
    plt.close()
    plt.figure()
    plt.plot(ship._logs["times"][:, 0], ship.actuators[0]._logs[:, :])
    plt.show()
    plt.close()
    plt.figure()
    plt.plot(ship._logs["times"][:, 0], ship.actuators[1]._logs[:, :])
    plt.show()

if __name__ == "__main__":
    test()