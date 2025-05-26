from nav_env.control.controller import ControllerBase, Command
from nav_env.control.states import States
import casadi as cd, numpy as np
from typing import Callable
from nav_env.control.path import Waypoints

"""
Base MPC is designed according to "Risk-BasedModelPredictiveControl for Autonomous Ship Emergency Management" by Simon Blindheim et al.



"""

class NMPC(ControllerBase):
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
        self.LBG = np.zeros(self.G.size()[0])
        self.UBG = np.zeros(self.G.size()[0])

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
            "lbx": np.concatenate([self.LBX, self.LBU]),
            "ubx": np.concatenate([self.UBX, self.UBU]),
            "lbg": self.LBG,
            "ubg": self.UBG
        }

        self.solver = cd.nlpsol("mpc_solver", "ipopt", nlp, options)
        solver_out = self.solver(**solver_in)
        arr = solver_out['x'].full().flatten()
        self.Xopt = arr[0:(self.horizon+1)*self.nx].reshape((self.nx, self.horizon+1))
        self.Uopt = arr[(self.horizon+1)*self.nx:].reshape((self.nu, self.horizon))

        # Remove last constraint (i.e. x[0] = x0) after solve is done, since it will change for next nlp
        self.G = self.G[0:-self.nx]
        self.LBG = self.LBG[0:-self.nx]
        self.UBG = self.UBG[0:-self.nx]

    def _set_x0_constraint(self, x0) -> None:
        self.G = cd.vertcat(
            self.G,
            self.X[0:self.nx] - x0
        )
        self.LBG = np.concatenate([self.LBG, np.zeros(self.nx)])
        self.UBG = np.concatenate([self.UBG, np.zeros(self.nx)])

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


def casadi_example() -> None:
    """
    Based on an example from https://vladimim.folk.ntnu.no/#/4 - updated to match actual casadi library 
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Decision variables
    x = cd.SX.sym("x", 3)
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
    
    # Objective function
    model = lambda x, u: (x * u + 0.5)*0.04 + x + 0.1*np.sin(x)
    lagrange = lambda x, u: x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + u[0]*u[0]
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
        
    print(nmpc)
    plt.plot(x_traj, label=[f"x{i+1}" for i in range(x_traj.shape[1])])
    plt.plot(u_traj, label="Command")
    plt.legend()
    plt.show()
    plt.close()


if __name__=="__main__":
    test()
    # casadi_example()
