from abc import ABC, abstractmethod
from typing import Any, Iterable
from dataclasses import fields
from nav_env.sensors.measurements import GNSSMeasurement, IMUMeasurement, BatteryMeasurement, RudderMeasurement, RPMMeasurement
from nav_env.sensors.noise import Noise


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
    def __init__(self, system:Any, src:str, *args, noise:Noise, id:int=None, **kwargs):
        self._system = system
        self._splitted_src = tuple([sub_str for sub_str in src.split('.') if len(sub_str)>0]) # allows to specify both .states.xy and states.xy with same outcomes
        self._noise = noise or Noise.default_noise(len(fields(self.valid_measurement_type())))
        self._id = id

    def get(self, *args, **kwargs) -> Any:
        obj = self._system
        for obj_path in self._splitted_src:
            obj = obj.__getattribute__(obj_path)
        if not isinstance(obj, Iterable):
            obj = (obj,)
        measurement = self.valid_measurement_type()(*obj)
        return self._noise(measurement)
    
    def __repr__(self) -> str:
        return f"{type(self).__name__} object (id:{self._id})"

    @abstractmethod
    def valid_measurement_type(self) -> Any:
        return None

class GNSS(Sensor):
    def __init__(self, system:Any, src:str, *args, noise:Noise=None, id:int=None, **kwargs):
        super().__init__(system=system, src=src, *args, noise=noise, id=id, **kwargs)

    def valid_measurement_type(self):
        return GNSSMeasurement

class IMU(Sensor):
    def __init__(self, system:Any, src:str, *args, noise:Noise=None, id:int=None, **kwargs):
        super().__init__(system=system, src=src, *args, noise=noise, id=id, **kwargs)

    def valid_measurement_type(self):
        return IMUMeasurement

class BatteryLevel(Sensor):
    def __init__(self, system:Any, src:str, *args, noise:Noise=None, id:int=None, **kwargs):
        super().__init__(system=system, src=src, *args, noise=noise, id=id, **kwargs)

    def valid_measurement_type(self):
        return BatteryMeasurement
    
class RudderIndicator(Sensor):
    def __init__(self, system:Any, src:str, *args, noise:Noise=None, id:int=None, **kwargs):
        super().__init__(system=system, src=src, *args, noise=noise, id=id, **kwargs)

    def valid_measurement_type(self):
        return RudderMeasurement
    
class RPM(Sensor):
    """
    Rotation Per Minute (RPM) Sensor, to be used for measuring rotation speed, e.g. of propellers
    """
    def __init__(self, system:Any, src:str, *args, noise:Noise=None, id:int=None, **kwargs):
        super().__init__(system=system, src=src, *args, noise=noise, id=id, **kwargs)

    def valid_measurement_type(self):
        return RPMMeasurement

def test() -> None:
    from nav_env.ships.ship import Ship

    src = 'states'
    
    
    os = Ship()
    # print(os.__getattribute__('states').__getattribute__('xy'))
    imu = IMU(os, '.states', id=1, noise=Noise(6*[{"distribution":"normal", "hyper":{"loc":5, "scale":1}}]))
    motor1 = RPM(os, 'states.x', id=2, noise=Noise(1*[{"distribution":"normal", "hyper":{"loc":-1, "scale":0.01}}]))
    print(imu._splitted_src, motor1._splitted_src)
    print(imu, motor1)
    print(imu.get(), motor1.get())
    os.states.x = 3.4
    os.states.y = 10.4
    print(imu.get(), motor1.get())


if __name__ == "__main__": test()



