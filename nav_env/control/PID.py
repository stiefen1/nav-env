from nav_env.control.controller import ControllerBase
from nav_env.ships.states import States3
from nav_env.control.command import GeneralizedForces
import numpy as np
from typing import Any


class PID(ControllerBase):
    """
    Control law is u = -(kp*e+kd*dedt+ki*I)
    """
    def __init__(self, kp, ki=None, kd=None, dt:float=None, anti_windup:tuple=None):
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


    def get(self, x, xd) -> Any:
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
        
        e = x-xd # Error
        dedt = (e - self._prev_e)/self._dt # Derivative of the error
        self._prev_e = e
        self._integral = np.clip(self._integral + e * self._dt, -self._anti_windup, self._anti_windup) # Integral with antiwindup
        u = -(self._kp @ e + self._ki @ self._integral + self._kd @ dedt)
        return u
    
    def reset(self) -> None:
        self._integral = 0
        self._prev_e = np.zeros(shape=(self._nx, 1))
    

class HeadingAndSpeedController(PID):
    def __init__(self, pid_gains_heading:tuple, pid_gains_speed:tuple, dt:float=None, anti_windup:tuple=None):
        assert len(pid_gains_heading) == 3, f"pid_gains_heading must have length 3, not {len(pid_gains_heading)}"
        assert len(pid_gains_speed) == 3, f"pid_gains_speed must have length 3, not {len(pid_gains_speed)}"
        kp = (pid_gains_heading[0], pid_gains_speed[0])
        ki = (pid_gains_heading[1], pid_gains_speed[1])
        kd = (pid_gains_heading[2], pid_gains_speed[2])
        super().__init__(kp, ki=ki, kd=kd, dt=dt, anti_windup=anti_windup)

    def get(self, x:States3, xd:States3) -> GeneralizedForces:
        x = np.array([x.psi_deg, x.x_dot])
        xd = np.array([xd.psi_deg, xd.x_dot])
        u = super().get(x, xd)
        # print("x, xd: ", x, xd)
        return GeneralizedForces(f_x=u[1, 0], tau_z=u[0, 0])

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