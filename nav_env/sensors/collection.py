from nav_env.sensors.sensors import Sensor, GNSS, IMU, BatteryLevel, RudderIndicator, RPM
from typing import Any
from nav_env.actuators.collection import get_all_objects_of_type_in_iterable


class SensorCollection:
    def __init__(self, sensors:list[Sensor]=None) -> None:
        self._sensors = sensors

    def get(self) -> list[Any]:
        measurements = []
        for sensor in self:
            measurements.append(sensor.get())
        return tuple(measurements)

    def append(self, actuator:Sensor) -> None:
        assert isinstance(actuator, Sensor), f"Obstacle must be an instance of Sensor not {type(actuator)}"
        return self._sensors.append(actuator)

    def remove(self, actuator:Sensor) -> None:
        return self._sensors.remove(actuator)
    
    def __getitem__(self, index: int) -> Sensor:
        return self._sensors[index]
    
    def __len__(self) -> int:
        return len(self._sensors)

    def __repr__(self):
        return f"SensorCollection({len(self._sensors)} sensors: {[a for a in self._sensors]})"

    def __iter__(self):
        for obs in self._sensors:
            yield obs
    
    @staticmethod
    def empty() -> "SensorCollection":
        return SensorCollection([])
    
    @property
    def gnss(self) -> "SensorCollection":
        return SensorCollection(get_all_objects_of_type_in_iterable(self, GNSS))
    
    @property
    def imu(self) -> "SensorCollection":
        return SensorCollection(get_all_objects_of_type_in_iterable(self, IMU))
    
    @property
    def battery_level(self) -> "SensorCollection":
        return SensorCollection(get_all_objects_of_type_in_iterable(self, BatteryLevel))
    
    @property
    def rudder_indicator(self) -> "SensorCollection":
        return SensorCollection(get_all_objects_of_type_in_iterable(self, RudderIndicator))
    
    @property
    def rpm(self) -> "SensorCollection":
        return SensorCollection(get_all_objects_of_type_in_iterable(self, RPM))
    

def test() -> None:
    pass
 
if __name__ == "__main__": test()
    

