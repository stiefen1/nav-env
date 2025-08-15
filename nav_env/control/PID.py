from nav_env.control.controller import Controller
from nav_env.ships.states import States3
from nav_env.control.command import GeneralizedForces
import numpy as np
from typing import Any, Callable
from nav_env.utils.math_functions import wrap_angle_to_pmpi_degrees
from nav_env.ships.physics import ShipPhysics


class PID(Controller):
    """
    Control law is u = -(kp*e+kd*dedt+ki*I)
    """
    def __init__(self, kp, *args, ki=None, kd=None, dt:float=None, anti_windup:tuple=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._dt = dt if dt is not None else 1
        self._kp:np.ndarray = gain_as_matrix(kp)
        self._ki:np.ndarray = gain_as_matrix(ki) if ki is not None else np.zeros_like(self._kp)
        self._kd:np.ndarray = gain_as_matrix(kd) if kd is not None else np.zeros_like(self._kp)
        assert len(self._kp.shape) == 2, f"Kp must be 2D but has shape {self._kp.shape}"
        assert len(self._ki.shape) == 2, f"Ki must be 2D but has shape {self._ki.shape}"
        assert len(self._kd.shape) == 2, f"Kd must be 2D but has shape {self._kd.shape}"

        self._nu:int = self._kp.shape[0]
        self._nx:int = self._kp.shape[1]

        if isinstance(anti_windup, float) or isinstance(anti_windup, int):
            anti_windup = (anti_windup,) 
        self._anti_windup = np.array(anti_windup) if anti_windup is not None else np.array(self._nx * [float('inf')])
        self._anti_windup = self._anti_windup[:, None]
        self.reset()


    def __get__(self, x, xd, sat_fn:Callable=lambda xin: xin) -> Any:
        """
        sat_fn aims at limiting the error, especially when states involves angles. For example if x-xd is an angle error, 
        it might be necessary to make sur this error remains in [-pi, pi] to avoid agressive manoveurs (e.g. ship making
        a full turn on itself instead of slightly modifying its course) 
        """
        # Convert to numpy array if siso
        if isinstance(x, float) or isinstance(x, int):
            x = np.array([x])
        if isinstance(xd, float) or isinstance(xd, int):
            xd = np.array([xd])

        # Convert to 2d object if numpy array
        if isinstance(x, np.ndarray):
            if len(x.shape) != 2:
                x = x[:, None]
        else:
            raise TypeError(f"x must be a numpy array but is a {type(x)} object")
        if isinstance(xd, np.ndarray):
            if len(xd.shape) != 2:
                xd = xd[:, None]
        else:
            raise TypeError(f"xd must be a numpy array but is a {type(xd)} object")

        assert x.shape[0] == self._nx and x.shape[1] == 1, f"x must have shape ({self._nx}, 1) but has shape {x.shape}"
        assert xd.shape[0] == self._nx and xd.shape[1] == 1, f"xd must have shape ({self._nx}, 1) but has shape {xd.shape}"

        self.save(x, xd)
        e = sat_fn(x-xd) # Error
        dedt = (e - self._prev_e)/self._dt # Derivative of the error
        self._prev_e = e
        self._prev_dedt = dedt
        self._integral = np.clip(self._integral + e * self._dt, -self._anti_windup, self._anti_windup) # Integral with antiwindup
        u = -(self._kp @ e + self._ki @ self._integral + self._kd @ dedt)
        # print(f"\t{x[0, 0]:.1f}\t{xd[0, 0]:.1f}\t{1e6*self._kp[0, 0]*e[0, 0]:.1f}\t{1e6*self._ki[0, 0]*self._integral[0, 0]:.1f}\t{1e6*self._kd[0, 0]*dedt[0, 0]:.1f}")
        return u
    
    def reset(self) -> None:
        self._integral = 0
        self._prev_e = np.zeros(shape=(self._nx, 1))
        self._prev_dedt = np.zeros(shape=(self._nx, 1))
        self._logs = {"x": np.zeros((0, self._nx)), "x_des": np.zeros((0, self._nx))}

    def save(self, x:np.ndarray, x_des:np.ndarray) -> None:
        self._logs["x"] = np.append(self._logs["x"], x.T, axis=0)
        self._logs["x_des"] = np.append(self._logs["x_des"], x_des.T, axis=0)

class HeadingAndSpeedController(PID):
    def __init__(self, pid_gains_heading:tuple, pid_gains_speed:tuple, *args, dt:float=None, anti_windup:tuple=None, ship_physics:ShipPhysics=None, wind_feedforward:bool=False, **kwargs):
        assert len(pid_gains_heading) == 3, f"pid_gains_heading must have length 3, not {len(pid_gains_heading)}"
        assert len(pid_gains_speed) == 3, f"pid_gains_speed must have length 3, not {len(pid_gains_speed)}"
        kp = (-pid_gains_heading[0], pid_gains_speed[0])
        ki = (-pid_gains_heading[1], pid_gains_speed[1])
        kd = (-pid_gains_heading[2], pid_gains_speed[2])
        self.last_commanded_force = None
        super().__init__(kp, *args, ki=ki, kd=kd, dt=dt, anti_windup=anti_windup, ship_physics=ship_physics, wind_feedforward=wind_feedforward, **kwargs)
        

    def __get__(self, x:States3, xd:States3, *args, **kwargs) -> GeneralizedForces:
        x = np.array([x.psi_deg, (x.x_dot**2+x.y_dot**2)**0.5])
        xd = np.array([xd.psi_deg, xd.x_dot])
        # print(x[1], xd[1])
        u = super().__get__(x, xd, sat_fn=lambda h_and_s: np.array((wrap_angle_to_pmpi_degrees(h_and_s[0]), h_and_s[1])))
        # print(f"Heading (des, actual): {xd[0]:.1f}, {x[0]:.1f}")
        # print(f"Speed (des, actual): {xd[1]:.1f}, {x[1]:.1f}")
        self.last_commanded_force = GeneralizedForces(f_x=u[1, 0], tau_z=u[0, 0])
        return GeneralizedForces(f_x=u[1, 0], tau_z=u[0, 0])
    
    # def save(self) -> None:


# class TrajectoryTrackingController(PID):


def gain_as_matrix(k:Any) -> np.ndarray:
    if isinstance(k, np.ndarray):
        if len(k.shape) == 2:
            return k
        elif len(k.shape) == 1:
            return k[None, :]
        else:
            raise ValueError(f"Gain is a numpy array with invalid shape {k.shape}")
    elif isinstance(k, tuple) or isinstance(k, list):
        return np.diag(k)
    elif isinstance(k, float) or isinstance(k, int):
        return np.array([[k]])
    else:
        return TypeError(f"Gain has invalid type {type(k)}")
    
def test_gain_as_matrix():
    print(gain_as_matrix(1.0))
    print(gain_as_matrix(1))
    print(gain_as_matrix(np.array([3, 2, ])))
    print(gain_as_matrix(np.array([[1, 0], [0, 10]])))
    print(gain_as_matrix((1, 2, 3.)))
    print(gain_as_matrix([2, 2, 3.]))

def test_pid_controller():
    import matplotlib.pyplot as plt

    A = 0.3*np.array([[0.8, -0.4], [0.5, 0.8]])
    B = 0.3*np.array([[1, 0], [0, 1]])
    x0 = np.array([0, 0])
    controller = PID(kp=(0.3, 0.3), ki=(10, 10), anti_windup=(10, 10), dt=0.1, kd=(1e-3, 1e-3))
    fig, ax = plt.subplots()
    ax.scatter(*x0, c='g')
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    plt.show(block=False)

    states = np.zeros((100, 2))
    xds = np.zeros_like(states)
    inputs = np.zeros((100, 2))


    x = x0.copy()[:, None]
    for i in range(100):
        states[i] = x.squeeze()
        xd = 40*np.array([np.cos(i/10), np.sin(i/10)])
        xds[i] = xd.squeeze()
        ax.cla()
        ax.scatter(*xd, c='r')
        u = np.clip(controller.get(x, xd).T, (-80, -80), (80, 80)).T
        print("u[0, 0]: ", u[0, 0])
        inputs[i] = u.squeeze()
        x = A@x + B@u
        
        ax.scatter(*x, c='black')
        ax.set_title(f"{i}/100")
        ax.set_xlim([-200, 200])
        ax.set_ylim([-200, 200])
        plt.pause(0.01)
        print(controller._integral)

    plt.figure()
    plt.plot(np.linspace(0, 99, 100)*0.1, states)
    plt.plot(np.linspace(0, 99, 100)*0.1, xds, '--')
    plt.plot(np.linspace(0, 99, 100)*0.1, inputs)
    plt.show()

def test_siso_pid_controller():
    import matplotlib.pyplot as plt

    A = np.array([0.3])
    B = np.array([0.3])
    x0 = np.array([0])
    controller = PID(kp=0.3, ki=10, anti_windup=10, dt=0.1, kd=1e-3)

    states = np.zeros((100, 1))
    xds = np.zeros_like(states)
    inputs = np.zeros((100, 1))


    x = x0.copy()
    for i in range(100):
        states[i] = x.squeeze()
        xd = 10 # np.array([1])#np.array([1 + np.sin(i/10)])
        xds[i] = xd
        u = controller.get(x, xd).T
        print(u)
        inputs[i] = u.squeeze()
        x = A@x + B@u
        
        plt.pause(0.01)
        print(controller._integral)

    plt.figure()
    plt.plot(np.linspace(0, 99, 100)*0.1, states)
    plt.plot(np.linspace(0, 99, 100)*0.1, xds, '--')
    plt.plot(np.linspace(0, 99, 100)*0.1, inputs)
    plt.show()
    

if __name__ == "__main__":
    # test_gain_as_matrix()
    test_pid_controller()
    test_siso_pid_controller()