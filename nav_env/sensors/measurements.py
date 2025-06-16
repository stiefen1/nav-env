from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class IMUMeasurement:
    ax:float = 0.0
    ay:float = 0.0
    az:float = 0.0
    ix:float = 0.0
    iy:float = 0.0
    iz:float = 0.0

    def __array__(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az, self.ix, self.iy, self.iz])

@dataclass
class GNSSMeasurement:
    north:float = 0.0
    east:float = 0.0

    def __array__(self) -> np.ndarray:
        return np.array([self.north, self.east])

@dataclass
class BatteryMeasurement:
    level:float = 0.0

    def __array__(self) -> np.ndarray:
        return np.array([self.level])

@dataclass
class RudderMeasurement:
    angle_deg:float = 0.0

    def __array__(self) -> np.ndarray:
        return np.array([self.angle_deg])

@dataclass
class RPMMeasurement:
    speed:float = 0.0

    def __array__(self) -> np.ndarray:
        return np.array([self.speed])

def test() -> None:
    imu_data = IMUMeasurement(0., 0., 0., 0., 0., 0.)

    print([val for val in imu_data.__dict__.values()])
if __name__ == "__main__": test()
    