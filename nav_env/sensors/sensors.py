from abc import ABC, abstractmethod
from typing import Any, Iterable
from dataclasses import fields
from nav_env.sensors.measurements import GNSSMeasurement, IMUMeasurement, BatteryMeasurement, RudderMeasurement, RPMMeasurement
from nav_env.sensors.noise import Noise
import numpy as np, warnings


"""
When building a system using these component I must be able to easily specify what is the system for each sensor in 
the system object. How to do that ?

"""

class Sensor(ABC):
    """
    To declare a new sensor object, one must simply declare a dataclass object, import it here and create a class that inherits
    from Sensor. The dataclass object to be used must be returned by the valid_measurement_type abstract method.

    To instantiate a new sensor object, one must specify the source object to be monitored and the "path" to the actual value 
    to be measured. For a measurement of the x pose of a ship, accessible through ship.pose.x, the system must be ship and the
    src parameter must be specified as 'pose.x' or similarly '.pose.x'
    
    """
    def __init__(self, src:str, *args, system:Any=None, noise:Noise=None, id:int=None, **kwargs):
        self._system = system
        self._splitted_src = tuple([sub_str for sub_str in src.split('.') if len(sub_str)>0]) # allows to specify both .states.xy and states.xy with same outcomes
        self.n = len(fields(self.valid_measurement_type()))
        self._noise = noise or Noise.default_noise(self.n)
        self._id = id
        self.reset()

    def reset(self) -> None:
        self._last_noisy_measurement = None
        self._last_clean_measurement = None
        self._logs = np.zeros((0, self.n))

    def get(self, *args, **kwargs) -> Any:
        obj = self._system
        for obj_path in self._splitted_src:
            obj = obj.__getattribute__(obj_path)
        if not isinstance(obj, Iterable):
            obj = (obj,)
        self._last_clean_measurement = self.valid_measurement_type()(*obj)
        self._last_noisy_measurement = self._noise(self._last_clean_measurement)
        return self._last_noisy_measurement
    
    def __repr__(self) -> str:
        return f"{type(self).__name__} object (id:{self._id})"
    
    def save(self, *args, **kwargs) -> Any:
        if self._last_noisy_measurement is not None:
            self._logs = np.append(self._logs, np.array(self._last_noisy_measurement).reshape(1, self.n), axis=0)
        else:
            warnings.warn(f"Last noisy measurement of {self} is None - save aborted")

    @abstractmethod
    def valid_measurement_type(self) -> Any:
        return None
    
    @property
    def system(self) -> Any:
        return self._system

    @system.setter
    def system(self, value) -> None:
        self._system = value

class GNSS(Sensor):
    def __init__(self, *args, src:str='.states.xy', system:Any=None, noise:Noise=None, id:int=None, **kwargs):
        super().__init__(system=system, src=src, *args, noise=noise, id=id, **kwargs)

    def valid_measurement_type(self):
        return GNSSMeasurement

class IMU(Sensor):
    def __init__(self, *args, src:str='.acc', system:Any=None, noise:Noise=None, id:int=None, **kwargs):
        super().__init__(system=system, src=src, *args, noise=noise, id=id, **kwargs)

    def valid_measurement_type(self):
        return IMUMeasurement

class BatteryLevel(Sensor):
    def __init__(self, *args, src:str='.battery', system:Any=None, noise:Noise=None, id:int=None, **kwargs):
        super().__init__(system=system, src=src, *args, noise=noise, id=id, **kwargs)

    def valid_measurement_type(self):
        return BatteryMeasurement
    
class RudderIndicator(Sensor):
    def __init__(self, *args, src:str='.rudder.angle', system:Any=None, noise:Noise=None, id:int=None, **kwargs):
        super().__init__(system=system, src=src, *args, noise=noise, id=id, **kwargs)

    def valid_measurement_type(self):
        return RudderMeasurement
    
class RPM(Sensor):
    """
    Rotation Per Minute (RPM) Sensor, to be used for measuring rotation speed, e.g. of propellers
    """
    def __init__(self, *args, src:str='propeller.rpm', system:Any=None, noise:Noise=None, id:int=None, **kwargs):
        super().__init__(system=system, src=src, *args, noise=noise, id=id, **kwargs)

    def valid_measurement_type(self):
        return RPMMeasurement

def test() -> None:
    from nav_env.ships.ship import MovingShip
    from nav_env.ships.states import States3
    
    os = MovingShip(states=States3(u=3, v=0.1))
    # print(os.__getattribute__('states').__getattribute__('xy'))
    imu = IMU(src='.states', system=os, id=1, noise=Noise(6*[{"distribution":"normal", "hyper":{"loc":5, "scale":1}}]))
    motor1 = RPM(src='states.x', system=os, id=2, noise=Noise(1*[{"distribution":"normal", "hyper":{"loc":-1, "scale":0.01}}]))
    print(imu._splitted_src, motor1._splitted_src)
    print(imu, motor1)
    print(imu.get(), motor1.get())
    os.states.x = 3.4
    os.states.y = 10.4
    print(imu.get(), motor1.get())

def test_sensor_as_part_of_ship() -> None:
    from nav_env.ships.ship import MovingShip
    from nav_env.ships.states import States3
    
    imu = IMU(src='.states', id=1, noise=Noise(6*[{"distribution":"normal", "hyper":{"loc":5, "scale":1}}]))
    motor1 = RPM(src='states.x', id=2, noise=Noise(1*[{"distribution":"normal", "hyper":{"loc":-1, "scale":0.01}}]))
    os = MovingShip(states=States3(u=3, v=0.1), sensors=[imu, motor1])    
    print(os.sensors.get())   

def test_gnss() -> None:
    from nav_env.ships.ship import MovingShip
    from nav_env.ships.states import States3
    
    gnss1 = GNSS(id=0, noise=Noise(2*[{"distribution":"normal", "hyper":{"loc":0, "scale":0.01}}]))
    os = MovingShip(states=States3(u=3, v=0.1), sensors=[gnss1])    
    print(os.sensors.get())

def test_gnss_in_sim() -> None:
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
    from nav_env.control.allocation import NonlinearControlAllocation

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
            (33, 0), 0, (-180, 180), (0, 300), dt
        ),
        AzimuthThruster(
            (-33, 0), 0, (-180, 180), (0, 300), dt
        )
            ])

    
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
            pid_gains_speed=(8e4, 1e4, 0),
            dt=dt,
            allocation=NonlinearControlAllocation(actuators=actuators)
        ),
        actuators=actuators,
        name=name,
        sensors=[GNSS(noise=Noise(2*[{"distribution":"normal", "hyper":{"loc":0.1, "scale":1}}])), GNSS(noise=Noise(2*[{"distribution":"normal", "hyper":{"loc":5, "scale":0.1}}]))]
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

    tf = 50
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
        v = np.linalg.norm(ship.states.xy_dot)
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
    plt.plot(ship._logs["times"][:, 0], ship.sensors[0]._logs[:, :])
    plt.show()  
    plt.close()
    plt.figure()
    plt.plot(ship.sensors[0]._logs[:, 0], ship.sensors[0]._logs[:, 1])
    plt.show()   
    plt.close()
    plt.figure()
    plt.plot(ship.sensors[1]._logs[:, 0], ship.sensors[1]._logs[:, 1])
    plt.show()   
    plt.close()




if __name__ == "__main__": test_gnss_in_sim() #test_gnss() # test_sensor_as_part_of_ship()



