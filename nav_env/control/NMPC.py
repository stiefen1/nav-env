from nav_env.control.controller import Controller, Command
from nav_env.control.states import States
import casadi as cd, numpy as np
from typing import Callable, Any, Union
from nav_env.control.path import Waypoints
from nav_env.actuators.collection import ActuatorCollection
from nav_env.wind.wind_vector import WindVector
from nav_env.ships.ship import States3
from nav_env.simulation.integration import Integrator, Euler
import warnings
from copy import deepcopy

# TODO: Implement NMPC controller from this paper: MPC-based Mid-level Collision Avoidance for ASVs using Nonlinear Programming (https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2479486/CCTA17_0172_FI.pdf?sequence=2)

WINDX = 0
class NMPC(Controller):
    def __init__(
            self,
            lagrange:Callable,
            mayer:Callable,
            model:Callable,
            lbu:tuple,
            ubu:tuple,
            lbx:tuple=None,
            ubx:tuple=None,
            horizon:int=20,
            nx:int=3,
            ) -> None:
        self.model = model
        self.lagrange = lagrange
        self.mayer = mayer
        self.nx = nx

        assert len(lbu) == len(ubu), f"lbu and ubu must have same length but len(lbu) = {len(lbu)} != {len(ubu)} = len(ubu)"
        self.lbu = np.array(lbu, dtype=float)
        self.ubu = np.array(ubu, dtype=float)

        if lbx is None:
            self.lbx = np.full(self.nx, -np.inf)
        else:
            assert len(lbx) == self.nx, f"lbx must have length {self.nx} but len(lbx)={len(lbx)}"
            self.lbx = np.array(lbx, dtype=float)

        if ubx is None:
            self.ubx = np.full(self.nx, np.inf)
        else:
            assert len(ubx) == self.nx, f"ubx must have length {self.nx} but len(ubx)={len(ubx)}"
            self.ubx = np.array(ubx, dtype=float)

        assert horizon > 0 and isinstance(horizon, int), f"horizon must be an integer > 0 but is {horizon}"
        self.horizon = horizon
        self._init_nlp()

    def _init_nlp(self) -> None:
        self.X = cd.SX.sym("X", self.nx*(self.horizon+1))
        self.U = cd.SX.sym("U", self.nu*self.horizon)
        self.G = cd.SX.sym("G", 0, 0)
        self.acc_cost = cd.SX(0)

        # Numeric bounds
        self.LBX = []
        self.UBX = []
        self.LBU = []
        self.UBU = []

        for k in range(self.horizon):
            self.G = cd.vertcat(
                self.G,
                self.X[(k+1)*self.nx:(k+2)*self.nx] - self.model(self.X[k*self.nx:(k+1)*self.nx], self.U[k*self.nu:(k+1)*self.nu])
            )
            self.LBU.extend(self.lbu)
            self.UBU.extend(self.ubu)
            self.LBX.extend(self.lbx)
            self.UBX.extend(self.ubx)
            self.acc_cost += self.lagrange(self.X[k*self.nx:(k+1)*self.nx], self.U[k*self.nu:(k+1)*self.nu])

        # Add terminal state bounds
        self.LBX.extend(self.lbx)
        self.UBX.extend(self.ubx)

        self.acc_cost += self.mayer(self.X[-self.nx::])
        self.LBG = self.G.size()[0] * [0.0]
        self.UBG = self.G.size()[0] * [0.0]

        self.Xopt = None
        self.Uopt = None

    def get_initial_guess(self, x0):
        # Initial guess based on random input commands
        X0 = list(x0)
        x_prev = np.array(x0, dtype=float)
        U0 = []
        for k in range(self.horizon):
            uk = np.random.sample(self.nu)*(self.ubu - self.lbu) + self.lbu
            xkp1 = np.array(self.model(x_prev, uk)).flatten()
            X0.extend(xkp1)
            U0.extend(uk)
            x_prev = xkp1
        return np.array(X0 + U0)

    def _solve_nlp(self, initial_guess, options:dict={'ipopt.print_level':0, 'print_time':0}) -> None:
        nlp = {
            "x": cd.vertcat(self.X, self.U),
            "f": self.acc_cost,
            "g": self.G
        }

        solver_in = {
            "x0": initial_guess,
            "lbx": self.LBX + self.LBU,
            "ubx": self.UBX + self.UBU,
            "lbg": self.LBG,
            "ubg": self.UBG
        }

        self.solver = cd.nlpsol("mpc_solver", "ipopt", nlp, options)
        solver_out = self.solver(**solver_in)
        arr = solver_out['x'].full().flatten()
        self.Xopt = arr[0:(self.horizon+1)*self.nx].reshape((self.horizon+1, self.nx)).T
        self.Uopt = arr[(self.horizon+1)*self.nx:].reshape((self.horizon, self.nu)).T

        # Remove last constraint (i.e. x[0] = x0) after solve is done, since it will change for next nlp
        self.G = self.G[0:-self.nx]
        self.LBG = self.LBG[0:-self.nx]
        self.UBG = self.UBG[0:-self.nx]

    def _set_x0_constraint(self, x0) -> None:
        self.G = cd.vertcat(
            self.G,
            self.X[0:self.nx] - x0
        )
        self.LBG.extend(self.nx*[0.0])
        self.UBG.extend(self.nx*[0.0])

    def get(self, states:States, desired_states:States, initial_guess: Waypoints=None, *args, **kwargs) -> Command:
        # Add initial constraint X[:, 0] = x0
        self._set_x0_constraint(states)
        if initial_guess is None:
            X0 = self.get_initial_guess(states)
        self._solve_nlp(X0, *args, **kwargs)
        return self.u0
    
    def reset(self) -> None:
        pass

    def __str__(self) -> str:
        out = f"{type(self).__name__}("
        for key in ["model", "mayer", "lagrange", "lbx", "ubx", "lbu", "ubu"]:
            out += f"\n\t{key}: {self.__dict__[key]}"
        return out + "\n)"

    
    @property
    def ng(self) -> int:
        return self.G.size()[0]
    
    @property
    def nu(self) -> int:
        return len(self.lbu)
    
    @property
    def u0(self) -> tuple:
        """
        Returns first command input u0
        """
        if self.Uopt is None:
            return None
        return tuple(self.Uopt[0:self.nu, 0].tolist())

class NMPCPathTracking(Controller):
    """
    Design according to "Risk-BasedModelPredictiveControl for Autonomous Ship Emergency Management" by Simon Blindheim et al: https://www.sciencedirect.com/science/article/pii/S2405896320318681
    Key characteristics:

    - Direct Multiple-shooting


    Inputs:
        - Reference path: path (piece-wise linear functions [x(.), y(.)])
            --> Path progression is parametrized by a decision variable alpha >= 0
            --> We want this path progression to increase by alpha_step at each time-step (i.e. constant speed along the path)
            --> Cost function associated to path progression is:

                kappa^T \cdot [
                                || r(alpha_k) - p_k ||_2**2                         Minimize error between ship position and path at timestep k
                                || alpha_k - alpha_{k-1} - alpha_step ||_2**2       Make path progression constant along the path
                                beta_k                                              Minimize speed error w.r.t s_ref                                      
                                ]


        - Reference speed: s_ref
            --> constraint for penalizing speed larger than reference: u**2 + v**2 <= s_ref**2 + beta, beta>=0

    """
    def __init__(
            self,
            route:Waypoints,
            physics:Any,
            actuators:Union[ActuatorCollection, list],
            weights:dict,
            # speed_ref:float,
            # alpha_step:float, # dt is not known here, so alpha_step must be provided
            integrator:Integrator=None,
            dt:float=None,
            horizon:int=20

    ) -> None:
        super().__init__(actuators=actuators)
        self.route = route
        self.physics = physics
        # self.actuators = actuators if isinstance(actuators, ActuatorCollection) else ActuatorCollection(actuators)
        self.weights = weights
        # self.s_ref = speed_ref
        # self.a_step = alpha_step
        self.horizon = horizon
        self.nx = 6

        if integrator is not None:
            self.integrator = integrator
        elif dt is not None:
            self.integrator = Euler(dt)
        else:
            self.integrator = Euler()
            warnings.warn(f"Neither dt or integrator was specified. Default to Euler integration with dt={self.dt}")

        self.reset()

    @property
    def dt(self) -> float:
        return self.integrator.dt

    def reset(self) -> None:
        self._init_nlp()

    def model(self, x:Any, u:Any, wind:WindVector, use_casadi:bool=True) -> np.ndarray:
        control_forces = self.copy_of_actuators.dynamics(u, use_casadi=use_casadi, do_clip=False)
        if use_casadi:
            x = States3(*[x[i, 0] for i in range(x.size()[0])])
        derivative, _ = self.physics.get_time_derivatives_and_forces(x, wind=wind, control_forces=control_forces, input_uvr_in_ship_frame=True, get_uvr_dot_in_ship_frame=True, get_uvr_in_ship_frame=True, use_casadi=use_casadi)
        
        # when we say "get_uvr_in_ship_frame" it means:
        # -->   x_dot_dot, y_dot_dot, psi_dot_dot are in ship frame

        # -->   x_dot, y_dot, psi_dot are simply equal to the input, here in ship frame
        # -->   It means that x, y, psi are in ship frame as well
        # -->   but path is in world, so we cannot use it for the cost function
        
        new_states, _ = self.integrator(x, derivative)
        return new_states.to_casadi() if use_casadi else new_states

    def _init_stage_cost(self) -> None:
        """
        Init Xi and Epsilon cost function (as lambda functions)
        """
        assert 'kappa' in self.weights, f"kappa not found in weights dictionnary"
        assert 'Lambda' in self.weights, f"Lambda not found in weights dictionnary"
        assert 'Delta' in self.weights, f"Delta not found in weights dictionnary"

        kappa:np.ndarray = self.weights['kappa']    # 3 x 1 matrix penalizing path tracking error
        Lambda:np.ndarray = self.weights['Lambda']  # nu x nu matrix penalizing control input
        Delta:np.ndarray = self.weights['Delta']    # nu x nu matrix penalizing control input rate (u_k - u_{k-1})

        # Init stage cost lambdas
        self.xi_stage_cost = lambda x_k, alpha_k, alpha_k_prev, alpha_step, beta_k, route=self.route: cd.mtimes(kappa[None, :], cd.vertcat(
            cd.dot(route.r(alpha_k) - x_k[0:2], route.r(alpha_k) - x_k[0:2]),
            x_k[4]**2 + 0.05*x_k[5]**2,# (alpha_k - alpha_k_prev - alpha_step)**2,
            beta_k#  + 1e2*x_k[4]**2
        ))
        self.epsilon_stage_cost_part_A = lambda u_k: cd.mtimes(cd.mtimes(cd.transpose(u_k), Lambda), u_k)
        self.epsilon_stage_cost_part_B = lambda u_k, u_k_prev: cd.mtimes(cd.mtimes(cd.transpose(u_k - u_k_prev), Delta), (u_k - u_k_prev))

    def _set_state_and_actuator_bounds(self) -> None:
        self.lbu = [cd.DM(ui) for ui in self.actuators.u_min]
        self.ubu = [cd.DM(ui) for ui in self.actuators.u_max]
        self.lbx = [-cd.inf, -cd.inf, -cd.inf, cd.DM(0.0), cd.DM(-1), -cd.inf] # self.nx * [-cd.inf]
        self.ubx = [cd.inf, cd.inf, cd.inf, cd.inf, cd.DM(1), cd.inf] # self.nx * [cd.inf]

    def _vertcat_decision_variables_and_bounds(self) -> None:
        self.DV = cd.vertcat( # Decision Variables
            self.X_sym,
            self.U_sym,
            self.beta_sym,
            self.alpha_sym,
            self.s_ref_sym
        )

        # Lower Bound of Decision Variables = Lower Bound X + Lower Bound U + Lower Bound Beta + Lower Bound Alpha
        self.LBDV =  cd.DM(self.LBX + self.LBU + self.LBB + self.LBA + [cd.DM(0.0)])

        # Upper Bound of Decision Variables = Upper Bound X + Upper Bound U + Upper Bound Beta + Upper Bound Alpha
        self.UBDV = cd.DM(self.UBX + self.UBU + self.UBB + self.UBA + [cd.inf])

    def _init_optimization_variables(self) -> None:
        self.X_sym = cd.SX.sym("X", self.nx*(self.horizon+1))       # X = [eta, nu] = [N, E, psi, u, v, r]
        self.U_sym = cd.SX.sym("U", self.actuators.nu*self.horizon)
        self.beta_sym = cd.SX.sym("beta", self.horizon)     
        self.alpha_sym = cd.SX.sym("alpha", self.horizon+1)
        self.G_sym = cd.SX.sym("G", 0, 0)
        self.s_ref_sym = cd.SX.sym("s_ref", 1, 1)
        self.acc_cost = cd.SX(0) # accumulated cost

        # Numeric bounds
        self.LBX = []
        self.UBX = []
        self.LBU = []
        self.UBU = []
        self.LBB = []
        self.UBB = []
        self.LBA = []
        self.UBA = []
        self.LBG = []
        self.UBG = []

    def _set_epsilon_cost(self) -> None:
        # Input cost
        for k in range(0, self.horizon):
            self.acc_cost += self.epsilon_stage_cost_part_A(self.U_sym[k*self.actuators.nu:(k+1)*self.actuators.nu])
            if k>0:
                self.acc_cost += self.epsilon_stage_cost_part_B(self.U_sym[k*self.actuators.nu:(k+1)*self.actuators.nu], self.U_sym[(k-1)*self.actuators.nu:k*self.actuators.nu])

    def _set_state_and_command_bounds(self) -> None:
        self.LBU = self.horizon * self.lbu
        self.UBU = self.horizon * self.ubu
        self.LBX = (self.horizon+1) * self.lbx
        self.UBX = (self.horizon+1) * self.ubx
        
    def _set_dynamics_constraints(self, wind:WindVector) -> None:
        ### Dynamics    AND    constraints on x, u    AND    constraints u**2 + v**2 <= s_ref**2 + beta    AND    beta >= 0
        self.copy_of_actuators = deepcopy(self.actuators) # We don't want to modify states of the actual actuators
        for k in range(self.horizon):
            # 0 <= x_k+1 - f(x_k, u_k) <= 0
            self.G_sym = cd.vertcat(
                self.G_sym,
                self.X_sym[(k+1)*self.nx:(k+2)*self.nx] - self.model(self.X_sym[k*self.nx:(k+1)*self.nx], self.U_sym[k*self.actuators.nu:(k+1)*self.actuators.nu], wind)
            )
            self.LBG.extend(self.nx * [cd.DM(0.0)])
            self.UBG.extend(self.nx * [cd.DM(0.0)])

    def _set_speed_constraints(self) -> None:
        #  u**2 + v**2 = s_ref**2 + beta            ##### ATTENTION CE N'EST PAS LA CONTRAINTE ORIGINAL DE SIMON -> JE VEUX UN BON TRACKING -> B EST UNE SLACK VARIABLE QU'ON MINIMISE
        for k in range(1, self.horizon+1):
            self.G_sym = cd.vertcat(
                self.G_sym,
                self.X_sym[k*self.nx+3]**2 + self.X_sym[k*self.nx+4]**2 - self.s_ref_sym**2 - self.beta_sym[k-1],
            )
            self.LBG.extend([cd.DM(0.0)])
            self.UBG.extend([cd.DM(0.0)])

    def _set_alpha_and_beta_bounds(self, alpha_step:float) -> None:
        self.LBB = (self.horizon) * [cd.DM(0.0)]
        self.UBB = (self.horizon) * [cd.inf]
        self.LBA = []
        self.UBA = []
        for k in range(self.horizon+1):
            self.LBA.append(cd.DM(self._alpha_0 + k*alpha_step))
            self.UBA.append(cd.DM(self._alpha_0 + (k+1)*alpha_step))
        # self.LBA = (self.horizon+1) * [cd.DM(self._alpha_0)]
        # self.UBA = (self.horizon+1) * [cd.DM(1.0)]


    def _init_nlp(self) -> None:
        """
        Only called once when object is instantianted. Hence no initial condition is included here.
        """
        # Path progression variable
        self._alpha_0 = 0.0 # Initially, path progression is zero. It will be updated after each iteration

        self._init_stage_cost() # Declare lambda function
        self._set_state_and_actuator_bounds() # Set self.lbx, self.lbu, etc..
        self._init_optimization_variables() # Declare self.X_sym, self.U_sym, LBG, UBG, LBX, etc..
  
        self._set_speed_constraints() # u**2 + v**2 <= s_ref**2 + beta
        
        self._set_state_and_command_bounds()

        self._set_epsilon_cost()

        # Initial solutions are None until we call the get() method
        self.Xopt = None
        self.Uopt = None

    def _set_xi_cost(self, alpha_step:float) -> None:
        # Xi cost (path tracking)
        gamma = 1
        for k in range(1, self.horizon+1):
            self.acc_cost += gamma**k * self.xi_stage_cost(self.X_sym[(k)*self.nx:(k+1)*self.nx], self.alpha_sym[k], self.alpha_sym[k-1], alpha_step, self.beta_sym[k-1])

    def _set_constraints_before_optimization(self, x0:States3, x_des:States3, wind:WindVector, *args, **kwargs) -> None:
        """
        x, y, psi are ship's pose in world frame
        x_dot, y_dot, psi_dot are ship's speed in ship's frame
        """
        s_ref = x_des.x_dot # u (speed in surge)
        alpha_step = self.dt * s_ref / self.route.d_tot

        # Concatenate all decision variables in DV, LBDV, UBDV
        self._set_alpha_and_beta_bounds(alpha_step=alpha_step) # 1 >= alpha >= 0 AND beta >= 0
        self._vertcat_decision_variables_and_bounds()

        # Set dynamic constraints given wind perturbation
        self._set_dynamics_constraints(wind) # The reason why we do it here is because we want to be adaptive to wind variations -> Maybe include wind as DV and constraint it for optimized online performance

        # Reinitialize accumulated cost
        self.acc_cost = cd.SX(0) # accumulated cost
        self._set_epsilon_cost()
        self._set_xi_cost(alpha_step)
        
        # 0 <= x0 - x(0) <= 0
        self.G_sym = cd.vertcat(
            self.G_sym,
            self.DV[0:self.nx] - x0
        )
        self.LBG.extend(self.nx*[cd.DM(0.0)])
        self.UBG.extend(self.nx*[cd.DM(0.0)])

        # 0 <= s_ref - self.s_ref_sym <= 0
        self.G_sym = cd.vertcat(
            self.G_sym,
            self.DV[-1] - s_ref
        )
        self.LBG.extend([cd.DM(0.0)])
        self.UBG.extend([cd.DM(0.0)])

        # 0 <= alpha0 - alpha(0) <= 0
        self.G_sym = cd.vertcat(
            self.G_sym,
            self.DV[-1-self.horizon-1] - self._alpha_0
        )
        self.LBG.extend([cd.DM(0.0)])
        self.UBG.extend([cd.DM(0.0)])
    
    def __get__(self, states:States3, desired_states:States3, *args, initial_guess: Waypoints=None, wind: WindVector=None, **kwargs) -> ActuatorCollection:
        """
        states is in ship frame (obtained from gnc)
        """
        wind = wind or WindVector(states.xy, vector=(WINDX, 0.))
        self._set_constraints_before_optimization(states, desired_states, wind, *args, **kwargs) # alpha_0 IS FIRST 0 AND THEN WE UPDATE IT AFTER EACH TIME-STEP SINCE IT IS ONE OF THE NLP's OUTPUT
        if initial_guess is None:
            X0 = self.get_initial_guess(states, desired_states, wind, *args, **kwargs)
        self._solve_nlp(X0, *args, **kwargs)
        # Convert u0 into commands to be distributed accross actuators
        count = 0
        commands = []
        for a in self.actuators:
            command = a.valid_command(*self.u0[count:count+a.nu])
            commands.append(command)
            count += a.nu
        return commands
    
    def get_initial_guess(self, x0:States3, x_des:States3, wind:WindVector, *args, **kwargs) -> list:
        """
        Initial guess must be a list containing:

            Variable         |   Dimension   |  Computed according to
            -----------------|---------------|-------------------------
            self.X_sym,      |   nx*(h+1)    |  dynamics     
            self.U_sym,      |   nu*h        |  random within bounds
            self.beta_sym,   |   h           |  zeros
            self.alpha_sym,  |   h+1         |  alpha_step
            self.s_ref_sym   |   1           |  s_ref
        
        """
        s_ref = x_des.x_dot # u (speed in surge)
        alpha_step = self.dt * s_ref / self.route.d_tot

        ### Initial guess for x and u
        x_guess = [*x0.tolist()]
        u_guess = []
        x = x0
        # print("Initial guess: ")
        self.copy_of_actuators = deepcopy(self.actuators)
        for k in range(self.horizon):
            if self.Uopt is None or k>= self.horizon-1:
                u:list = self.copy_of_actuators.u_mean # [0, 150, 0, 150] # self.actuators.sample_within_bounds(as_list=True) # [0, 100, 0, 100] # self.actuators.u_max# 
            else:
                u:list = self.Uopt[:, k+1].tolist()
            x:States3 = self.model(x, u, wind, use_casadi=False)

            x_guess += x.tolist()
            u_guess += u

        ### Initial guess for beta
        beta_guess = (self.horizon)*[0.0]

        ### Initial guess for alpha
        alpha_guess = []
        alpha = self._alpha_0
        for _ in range(self.horizon+1):
            alpha_guess.append(alpha)
            alpha += alpha_step

        ### Initial guess for s_ref
        s_ref_guess = [s_ref]

        # print(len(x_guess),  len(u_guess),  len(beta_guess),  len(alpha_guess),  len(s_ref_guess))
        return x_guess + u_guess + beta_guess + alpha_guess + s_ref_guess

    def _solve_nlp(self, initial_guess, options:dict={'ipopt.print_level':0, 'print_time':False, 'ipopt.linear_solver': 'mumps', 'ipopt.max_iter':300}) -> None:
        # print(self.DV.size(), self.acc_cost.size(), self.G_sym.size(), len(initial_guess), self.LBDV.size(), self.UBDV.size(), len(self.LBG), len(self.UBG))
        
        nlp = {
            "x": self.DV,
            "f": self.acc_cost,
            "g": self.G_sym
        }

        solver_in = {
            "x0": initial_guess,
            "lbx": self.LBDV,
            "ubx": self.UBDV,
            "lbg": cd.vertcat(*self.LBG),
            "ubg": cd.vertcat(*self.UBG)
        }   

        self.solver = cd.nlpsol("mpc_solver", "ipopt", nlp, options)
        solver_out = self.solver(**solver_in)
        # print(solver_out['x'])
        arr = solver_out['x'].full().flatten()
        # g = solver_out['g'].full().flatten()
        # print(g.shape)

        # Declare it beforehand as it will never change
        idx_x0, idx_xf = 0, (self.horizon+1)*self.nx
        idx_u0, idx_uf = idx_xf, idx_xf+self.horizon*self.actuators.nu
        idx_b0, idx_bf = idx_uf, idx_uf+self.horizon
        idx_a0, idx_af = idx_bf, idx_bf+self.horizon+1


        self.Xopt = arr[idx_x0:idx_xf].reshape((self.horizon+1, self.nx)).T
        self.Uopt = arr[idx_u0:idx_uf].reshape((self.horizon, self.actuators.nu)).T
        self.Aopt = arr[idx_a0:idx_af]
        self.Bopt = arr[idx_b0:idx_bf]
        self.Sopt = arr[-1]


        self._alpha_0 = float(arr[-self.horizon-1]) # Retrieve next alpha for next iteration
        # print("new alpha zero: ", self._alpha_0)
        # print("Xopt: ", self.Xopt)
        # print("Uopt: ", self.Uopt)
        # print("Aopt: ", self.Aopt)
        # print("Bopt: ", self.Bopt)
        # print("Sopt: ", self.Sopt)

        # Remove constraints not relevant for next iterations
        # G is made of:

        """
            (h+1)   u**2 + v**2 - s_ref**2 - beta
            (nx*h)  dynamics
            (nx)    x0 - x(0)
            (1)     s_ref
            (1)     alpha0     
        """



        # Remove two last constraints (i.e. x[0] = x0 and alpha[0] = alpha0) after solve is done, since it will change for next nlp
        new_range = slice(0, -2-self.nx-self.nx*self.horizon)
        self.G_sym = self.G_sym[new_range]
        self.LBG = self.LBG[new_range]
        self.UBG = self.UBG[new_range]

    def eval_trajectory(self, x_des:States3) -> dict:
        assert self.Xopt is not None and self.Uopt is not None, f"No trajectory available in {self}, you must first call the get method"

        s_ref = x_des.x_dot
        alpha_step = self.dt * s_ref / self.route.d_tot

        # We do not consider weights here

        # xi
        ak_prev = None
        for k in range(self.horizon+1):
            xk = self.Xopt[:, k]
            ak = self.Aopt[k]
            if k > 0:
                bk = self.Bopt[k]
                stage_cost_k = self.xi_stage_cost(xk, ak, ak_prev, alpha_step, bk)
                

            if k < self.horizon:
                uk = self.Uopt[:, k]
                eps_A_k = self.epsilon_stage_cost_part_A(uk)
                uk_prev = uk

                if k > 0:
                    eps_B_k = self.epsilon_stage_cost_part_B(uk, uk_prev)
                

            

            ak_prev = ak
            

        # epsilon1

        # epsilon2



    def plot_trajectory(self, ax=None) -> None:
        ax.plot(self.Xopt[0, :], self.Xopt[1, :])

    def plot_desired_trajectory_from_optimization(self, ax=None) -> None:
        xs, ys = [], []
        for ak in self.Aopt:
            x, y = self.route(ak)
            xs.append(x)
            ys.append(y)
        # print("xy: ", xs, ys)
        ax.plot(xs, ys, '--b')

    @property
    def u0(self) -> tuple:
        """
        Returns first command input u0
        """
        if self.Uopt is None:
            return None
        return tuple(self.Uopt[0:self.actuators.nu, 0].tolist())
    
class OLD__NMPCPathTracking(NMPC):
    """
    Design according to "Risk-BasedModelPredictiveControl for Autonomous Ship Emergency Management" by Simon Blindheim et al: https://www.sciencedirect.com/science/article/pii/S2405896320318681
    Key characteristics:

    - Direct Multiple-shooting


    Inputs:
        - Reference path: path (piece-wise linear functions [x(.), y(.)])
            --> Path progression is parametrized by a decision variable alpha >= 0
            --> We want this path progression to increase by alpha_step at each time-step (i.e. constant speed along the path)
            --> Cost function associated to path progression is:

                kappa^T \cdot [
                                || r(alpha_k) - p_k ||_2**2                         Minimize error between ship position and path at timestep k
                                || alpha_k - alpha_{k-1} - alpha_step ||_2**2       Make path progression constant along the path
                                beta_k                                              Minimize speed error w.r.t s_ref                                      
                                ]


        - Reference speed: s_ref
            --> constraint for penalizing speed larger than reference: u**2 + v**2 <= s_ref**2 + beta, beta>=0
    """
    def __init__(
            self,
            route:Waypoints,
            model:Callable,
            actuators:ActuatorCollection,
            weights:dict,
            speed_ref:float,
            alpha_step:float, # dt is not known here, so alpha_step must be provided
            horizon:int=20,
    ) -> None:
        self._actuators = actuators
        lagrange, mayer, model, lbu, ubu, lbx, ubx = NMPCPathTracking.get_ready_for_nlp(route, model, actuators, weights, speed_ref, alpha_step, horizon)
        super().__init__(
            lagrange=lagrange,
            mayer=mayer,
            model=model,
            lbu=lbu,
            ubu=ubu,
            lbx=lbx,
            ubx=ubx,
            horizon=horizon,
            nx=3                # nx=6 even though psi is not minimized in any sense, it is still a decision variable
        )

    @staticmethod
    def get_ready_for_nlp(route:Waypoints, model:Callable, actuators:ActuatorCollection, weights:dict, speed_ref:float, alpha_step:float, horizon:int) -> tuple:
        """
        Convert problem into standard NLP format to be understandable by parent class NMPC
        """
        assert 'kappa' in weights, f"kappa not found in weights dictionnary"
        assert 'Lambda' in weights, f"Lambda not found in weights dictionnary"
        assert 'Delta' in weights, f"Delta not found in weights dictionnary"

        kappa = weights['kappa']    # 3 x 1 matrix penalizing path tracking error
        Lambda = weights['Lambda']  # nu x nu matrix penalizing control input
        Delta = weights['Delta']    # nu x nu matrix penalizing control input rate (u_k - u_{k-1})

        lbu = []
        ubu = []
        for a in actuators:
            lbu += list(a.u_min)
            ubu += list(a.u_max)

        
        lagrange = None
        mayer = None
        lbx = None
        ubx = None
        return lagrange, mayer, model, lbu, ubu, lbx, ubx
    
    def get(self, states:States, desired_states:States, initial_guess: Waypoints=None, *args, **kwargs) -> ActuatorCollection:
        u0 = super().get(states, desired_states, initial_guess=initial_guess, *args, **kwargs)

        # Convert u0 into commands to be distributed accross actuators
        count = 0
        commands = []
        for a in self._actuators:
            command = a.valid_command(*u0[count:count+a.nu])
            commands.append(command)
            count += a.nu
        return commands

def casadi_example() -> None:
    """
    Based on an example from https://vladimim.folk.ntnu.no/#/4 - updated to match actual casadi library 
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Decision variables
    x1:cd.SX = cd.SX.sym("x1", 1)
    x2 = cd.SX.sym("x2", 1)
    x3 = cd.SX.sym("x3", 1)
    # x = cd.SX.sym("x", 3)
    x = cd.vertcat(x1, x2, x3)
    print(x.size())
    

    # Parameters
    p = [5.00,1.00]

    # Objective function
    f = x[0]*x[0] + x[1]*x[1] + x[2]*x[2]

    # Concatenate nonlinear constraints
    g = cd.vertcat(
        6*x[0] + 3*x[1] + 2*x[2] - p[0],
        p[1]*x[0] +   x[1] -   x[2] -   1)
    
    print(f"f: {type(f)} size:{f.size()}, g: {type(g)} size:{g.size()}, x: {type(x)} size:{x.size()}")

    # Nonlinear bounds
    lbg = [0.00, 0.00]
    ubg = [0.00, 0.00]

    # Input bounds for the optimization variables
    lbx = [0.00, 0.00, 0.00]
    ubx = [ cd.inf,  cd.inf,  cd.inf]

    # Initial guess for the decision variables
    x0  = [0.15, 0.15, 0.00]

    # Create NLP solver
    nlp = {
        "x": x,
        "f": f,
        "g": g
    }

        # cd.NlpBuilder(cd.nlpIn(x=x),cd.nlpOut(f=f, g=g))
    solver = cd.nlpsol("mysolver", "ipopt", nlp)

    solver_in = {}
    # Pass the bounds and the initial values
    solver_in["x0"] = x0    # Optional
    solver_in["lbx"] = lbx
    solver_in["ubx"] = ubx
    solver_in["lbg"] = lbg
    solver_in["ubg"] = ubg

    for key, val in zip(nlp.keys(), nlp.values()):
        print(f"{key}: {val}")

    # Solve NLP
    solution:dict = solver(**solver_in)

    for key, val in zip(solution.keys(), solution.values()):
        print(f"{key}: {val} ({type(val)})")

    # Plotting
    x_opt = np.array(solution['x']).flatten()
    x2_opt = x_opt[2]
    x0_vals = np.linspace(0, 2, 100)
    x1_vals = np.linspace(0, 2, 100)
    X0, X1 = np.meshgrid(x0_vals, x1_vals)
    F = X0**2 + X1**2 + x2_opt**2

    # Constraint 1: 6*x0 + 3*x1 + 2*x2_opt - p[0] == 0
    C1 = 6*X0 + 3*X1 + 2*x2_opt - p[0]
    # Constraint 2: p[1]*x0 + x1 - x2_opt - 1 == 0
    C2 = p[1]*X0 + X1 - x2_opt - 1

    plt.figure(figsize=(8,6))
    # Plot cost function contours
    cs = plt.contour(X0, X1, F, levels=30, cmap='viridis')
    plt.clabel(cs, inline=1, fontsize=8)
    # Plot constraints
    plt.contour(X0, X1, C1, levels=[0], colors='red', linewidths=2, linestyles='--')
    plt.contour(X0, X1, C2, levels=[0], colors='blue', linewidths=2, linestyles='-.')
    # Plot optimal solution
    plt.plot(x_opt[0], x_opt[1], 'ro', markersize=10, label='Optimal solution')
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.title('Cost function contours and constraints (with $x_2$ fixed at optimal)')
    # plt.legend(['Constraint 1', 'Constraint 2', 'Feasible region', 'Optimal solution'])
    plt.legend()
    plt.grid(True)
    plt.show()

def test() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Objective function
    model = lambda x, u: (x * u + 0.5)*0.04 + x + 0.1*np.sin(x)
    # lagrange = lambda x, u: x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + u[0]*u[0]
    lagrange = lambda x, u: x.T @ np.array(3*[u])
    mayer = lambda xn: xn[0]*xn[0] + xn[1]*xn[1] + xn[2]*xn[2]
    nmpc = NMPC(
        lagrange,
        mayer,
        model,
        (-10,),
        (1,),
        horizon=30
    )
    
    x = np.array([1., -2., 0.])
    x_traj = np.ndarray((0, 3))
    u_traj = np.ndarray((0, 1))
    N = 200 # Simulate system for 100 timestamps
    for n in range(N):
        x_traj = np.append(x_traj, x[None], axis=0)
        u = nmpc.get(x, None)
        u_traj = np.append(u_traj, np.array([u]), axis=0)
        x = model(x, u)
        print(x[:, None].shape)
        cost = lagrange(x[:, None], u)
        print(cost)
        
    print(nmpc)
    plt.plot(x_traj, label=[f"x{i+1}" for i in range(x_traj.shape[1])])
    plt.plot(u_traj, label="Command")
    plt.legend()
    plt.show()
    plt.close()

def test_path() -> None:
    import matplotlib.pyplot as plt, numpy as np, os
    from nav_env.ships.simplified_physics import SimpleShipPhysics
    import nav_env.ships.physics as phy
    from nav_env.ships.ship import Ship
    from nav_env.control.path import Waypoints as WP
    from nav_env.actuators.actuators import AzimuthThrusterWithSpeed
    from nav_env.control.guidance import PathProgressionAndSpeedGuidance
    from nav_env.environment.environment import NavigationEnvironment
    from nav_env.wind.wind_source import UniformWindSource
    from math import pi, cos, sin

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
        (R*cos(-pi/4)+cx, R*sin(-pi/4)+cy)
    ], interp=INTERPOLATION)

    max_rate_rpm = 2 # = 2 * 2pi rad/min = 2 * 2pi / 60 rad/sec
    max_rate_rad_sec = max_rate_rpm * 2 * pi / 60
    actuators = [
                AzimuthThrusterWithSpeed(
                    (33, 0), 0, (-max_rate_rad_sec, max_rate_rad_sec), (0, 300), dt
                ),
                AzimuthThrusterWithSpeed(
                    (-33, 0), 0, (-max_rate_rad_sec, max_rate_rad_sec), (0, 300), dt
                )
            ]

    path_to_params = os.path.join('nav_env', 'ships', 'blindheim_risk_2020.json')

    own_ship = Ship(
        states=States3(x=wpts[0][0], y=wpts[0][1], x_dot=0.5, y_dot=2.7, psi_deg=-20),
        physics=phy.ShipPhysics(path_to_params),
        actuators=actuators,
        guidance=PathProgressionAndSpeedGuidance(
            wpts, 3
        ),
        controller=NMPCPathTracking(
            route=wpts,
            physics=SimpleShipPhysics(path_to_params),
            actuators=actuators,
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

    env = NavigationEnvironment(
        own_ships=[own_ship],
        dt=dt,
        wind_source=UniformWindSource(WINDX, 0)
    )


    lim = ((-100, -100), (2*R+100, R+100))
    ax = env.plot(lim)
    plt.show(block=False)
    x, y = [], []

    tf = 1000
    for t in np.linspace(0, tf, int(tf//dt)):
        ax.cla()
        wpts.scatter(ax=ax, color='black')
        # ax.scatter(*ship._gnc._guidance.current_waypoint, c='red')
        ax.set_title(f"{t:.2f}")
        env.step()
        v = np.linalg.norm(own_ship.states.xy_dot)
        # print("speed norm: ", v)
        if t%10 > 0:
            x.append(own_ship.states.x)
            y.append(own_ship.states.y)
        ax.plot(x, y, '--r')
        env.plot(lim, ax=ax)
        own_ship._gnc._controller.plot_trajectory(ax=ax)
        own_ship._gnc._controller.plot_desired_trajectory_from_optimization(ax=ax)
        ax.set_aspect('equal')
        plt.pause(1e-9)

    plt.close()

    plt.figure()
    plt.title("u(t)")
    plt.plot(own_ship.logs["times"], np.array(own_ship.logs["states"])[:, 3])
    plt.ylim([0, 5])
    plt.show()
    plt.close()

    plt.figure()
    plt.title("Propeller speed")
    plt.plot(own_ship.logs["times"], np.array(own_ship.logs["commands"])[:, 1])
    # print(np.array(own_ship.logs["commands"]).shape)
    plt.plot(own_ship.logs["times"], np.array(own_ship.logs["commands"])[:, 3])
    plt.ylim([0, 320])
    plt.show()
    plt.close()

    plt.figure()
    plt.title("Azimuth rate")
    plt.plot(own_ship.logs["times"], np.array(own_ship.logs["commands"])[:, 0])
    # print(np.array(own_ship.logs["commands"]).shape)
    plt.plot(own_ship.logs["times"], np.array(own_ship.logs["commands"])[:, 2])
    plt.ylim([-1, 1])
    plt.show()
    plt.close()

if __name__=="__main__":
    # test()
    test_path()
    # casadi_example()
