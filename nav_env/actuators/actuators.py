from abc import ABC, abstractmethod
from nav_env.control.command import GeneralizedForces
from math import pi, cos, sin

class Actuator(ABC):
    def __init__(self,
                 xy:tuple,
                 angle:float,
                 u_min:tuple | float | int,
                 u_max:tuple | float | int,
                 f_min:GeneralizedForces,
                 f_max:GeneralizedForces,
                 *args, **kwargs):
        """
        xy: Tuple containing x, y position w.r.t to cdg
        angle: Actuator's orientation in degrees
        u_min, u_max: Minimum and Maximum input commands (e.g. propeller's speed)
        f_min, f_max: Minimum and Maximum output generalized forces
        """
        if isinstance(u_min, float|int):
            u_min = (u_min,)
        if isinstance(u_max, float|int):
            u_max = (u_max,)
        assert len(u_min) == len(u_max), f"min and max input values must have the same length but are {len(u_min)} != {len(u_max)}"
        assert len(xy) == 2, f"The actuator's position must be a 2D tuple but has {len(xy)} dimensions"

        if f_min is None:
            f_min = -GeneralizedForces.inf()
        if f_max is None:
            f_max = GeneralizedForces.inf()

        self._xy = xy
        self._angle_deg = angle
        self._u_min = u_min
        self._u_max = u_max
        self._f_min = f_min
        self._f_max = f_max

    def __repr__(self):
        return f"{type(self).__name__} object at {self.xy}, {self.angle_deg} deg."
    

    @abstractmethod
    def dynamics(self, command:tuple, vr:tuple) -> GeneralizedForces:
        """
        vr: ship's relative speed to water current, required for computing the resulting generalized force
        Return generalized force based on command, which can be propeller's speed for instance.
        An actuator can consider different level of dynamics (e.g. the desired value is too further
        away from the current position, we can either consider the actuator has infinite authority or
        it is limited)
        """
        return GeneralizedForces()
        
    @property
    def nu(self) -> int:
        return len(self._u_min)
    
    @property
    def angle_rad(self) -> float:
        return self._angle_deg * pi / 180.0
    
    @property
    def u_max(self) -> tuple:
        return self._u_max
    
    @property
    def u_min(self) -> tuple:
        return self._u_min
    
    @property
    def f_min(self) -> GeneralizedForces:
        return self._f_min
    
    @property
    def f_max(self) -> GeneralizedForces:
        return self._f_max
    
    @property
    def xy(self) -> tuple:
        return self._xy
    
    @property
    def angle_deg(self) -> float:
        return self._angle_deg
    
    @property
    def lx(self) -> float:
        return self._xy[0]
    
    @property
    def ly(self) -> float:
        return self._xy[1]
    
    @property
    def r(self) -> tuple[float, float, float]:
        return (*self._xy, 0)

class Rudder(Actuator):
    """
    Aft rudders: Rudders are the primary steering device for conventional marine craft. They are located
    aft of the craft and the rudder force Fy will be a function of the rudder deflection (the drag force in the
    x direction is usually neglected in the control analysis). A rudder force in the y direction will produce
    a yaw moment that can be used for steering control.
    """
    def __init__(self,
                 xy:tuple,
                 angle:float,
                 u_min:tuple,
                 u_max:tuple,
                 *args,
                 f_min:GeneralizedForces=None,
                 f_max:GeneralizedForces=None,
                 angle_0:float=0,
                 **kwargs):
        super().__init__(xy, angle, u_min, u_max, f_min, f_max, *args, **kwargs)
        self._angle = angle_0

    def dynamics(self, command:tuple, vr:tuple) -> GeneralizedForces:
        """
        command: (Rudder's angle)
        """
        return GeneralizedForces()
    
class ControlSurface(Actuator):
    """
    Control surfaces: Control surfaces can be mounted at different locations to produce lift and drag forces.
    """
    def __init__(self,
                 xy:tuple,
                 angle:float,
                 u_min:tuple,
                 u_max:tuple,
                 *args,
                 f_min:GeneralizedForces=None,
                 f_max:GeneralizedForces=None,
                 angle_0:float=0,
                 **kwargs):
        super().__init__(xy, angle, u_min, u_max, f_min, f_max, *args, **kwargs)
        self._angle = angle_0

    def dynamics(self, command:tuple, vr:tuple) -> GeneralizedForces:
        """
        command: (control surface's angle)
        """
        return GeneralizedForces()
    
class WaterJet(Actuator):
    """
    Water jets: Water jets are an alternative to main propellers aft of the ship. They are usually used for high-speed craft.
    """
    def __init__(self,
                 xy:tuple,
                 angle:float,
                 u_min:tuple,
                 u_max:tuple,
                 *args,
                 f_min:GeneralizedForces=None,
                 f_max:GeneralizedForces=None,
                 angle_0:float=0,
                 **kwargs):
        super().__init__(xy, angle, u_min, u_max, f_min, f_max, *args, **kwargs)
        self._angle = angle_0

    def dynamics(self, command:tuple, vr:tuple) -> GeneralizedForces:
        """
        command: (control surface's angle)
        """
        return GeneralizedForces()
    

class AzimuthThruster(Actuator):
    def __init__(self,
                 xy:tuple,
                 angle:float, # Orientation when thruster is at its reference position
                 u_min:tuple,
                 u_max:tuple,
                 *args,
                 f_min:GeneralizedForces=None,
                 f_max:GeneralizedForces=None,
                 alpha_0:float=0.0, # Orientation w.r.t. initial orientation
                 speed_0:float=0.0,
                 **kwargs):
        super().__init__(xy, angle, u_min, u_max, f_min, f_max, *args, **kwargs)
        self._alpha: float = alpha_0
        self._speed: float = speed_0

    def dynamics(self, command:tuple, vr:tuple) -> GeneralizedForces:
        """
        command: (speed, angle in degrees)
        """
        alpha_deg = self._alpha + command[1]
        C_t = 2.2
        F_thrust = C_t * (command[0] * 60.0)**2

        ### Integrate actuator's position
        # u_sat = np.clip(command, u_min, u_max)
        # self._speed = u_sat[0]
        # self._angle = u_sat[1]

        ### Compute the resulting force based on relative

        return GeneralizedForces()
    
class Thruster(AzimuthThruster):
    """
    Special cases of Thruster:
        - Main Thrusters: The main propellers of the craft are mounted aft of the hull, usually in conjunction
        with rudders. They produce the necessary force Fx in the x direction needed for transit.
        - Tunnel Thrusters: These are transverse thrusters going through the hull of the craft. The propeller unit
        is mounted inside a transverse tube and produces a force Fy in the y direction. Tunnel thrusters are
        only effective at low speeds, which limits their use to low-speed maneuvering and stationkeeping.
        - Azimuth Thrusters: Azimuth thrusters: Thruster units that can be rotated an angle about the z axis and produce two
        force components (Fx, Fy) in the horizontal plane are usually referred to as azimuth thrusters. They are
        usually mounted under the hull of the craft and the most sophisticated units are retractable.
    """
    def __init__(self,
                 xy:tuple,
                 angle:float,
                 u_min:tuple,
                 u_max:tuple,
                 *args,
                 f_min:GeneralizedForces=None,
                 f_max:GeneralizedForces=None,
                 speed_0:float=0.0,
                 **kwargs):
        super().__init__(xy, angle, u_min, u_max, f_min, f_max, *args, speed_0=speed_0, alpha_0=0, **kwargs)

    def dynamics(self, command:tuple, vr:tuple) -> GeneralizedForces:
        """
        command: (propeller's speed)
        """
        return super().dynamics((command[0], 0), vr)
    
def test() -> None:
    a1 = AzimuthThruster((0, 0), 10, (-1, -30), (1, 30), GeneralizedForces(), GeneralizedForces())
    a2 = Thruster((1., -1.), -5, (1.), (3.), GeneralizedForces(), GeneralizedForces())
    a3 = Rudder((-3, 0.), 0, -10, 10, angle_0=-3)
    print(a1, a2, a3)
    print(GeneralizedForces().clip(a3.f_min, a3.f_max))
    
if __name__ == "__main__":
    test()

