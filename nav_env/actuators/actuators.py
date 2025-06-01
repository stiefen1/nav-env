from abc import ABC, abstractmethod
from nav_env.control.command import GeneralizedForces, AzimuthThrusterCommand, ThrusterCommand, Command
from math import pi, cos, sin

class Actuator(ABC):
    def __init__(self,
                 xy:tuple,
                 angle:float,
                 u_min:tuple | float | int,
                 u_max:tuple | float | int,
                 f_min:float,
                 f_max:float,
                 dt:float,
                 *args,
                 u_rate_min: tuple | float | int = None,
                 u_rate_max: tuple | float | int = None,
                 **kwargs
            ):
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
        
        if u_rate_min is None:
            u_rate_min = tuple(len(u_min) * [float('inf')])
        elif isinstance(u_rate_min, float|int):
            u_rate_min = (u_rate_min, )
        
        if u_rate_max is None:
            u_rate_max = tuple(len(u_max) * [float('inf')])
        elif isinstance(u_rate_max, float|int):
            u_rate_max = (u_rate_max, )

        assert len(u_min) == len(u_max), f"min and max input values must have the same length but are {len(u_min)} != {len(u_max)}"
        assert len(u_rate_min) == len(u_min), f"min input rate value has invalid length {len(u_rate_min)} != {len(u_min)}"
        assert len(u_rate_max) == len(u_max), f"max input rate value has invalid length {len(u_rate_max)} != {len(u_max)}"
        assert len(xy) == 2, f"The actuator's position must be a 2D tuple but has {len(xy)} dimensions"
        assert dt > 0.0, f"Sampling time must be greater than 0 but is {dt}"

        if f_min is None:
            f_min = -float('inf')
        if f_max is None:
            f_max = float('inf')

        self._xy = xy
        self._angle_deg = angle
        self._u_min = u_min
        self._u_max = u_max
        self._f_min = f_min
        self._f_max = f_max
        self._dt = dt
        self._u_rate_min = u_rate_min
        self._u_rate_max = u_rate_max

    def __repr__(self):
        return f"{type(self).__name__} object at {self.xy}, {self.angle_deg} deg."
    
    def dynamics(self, command:Command, *args, v_r:tuple=None, **kwargs) -> GeneralizedForces:
        """
        vr: ship's relative speed to water current, required for computing the resulting generalized force
        Return generalized force based on command, which can be propeller's speed for instance.
        An actuator can consider different level of dynamics (e.g. the desired value is too further
        away from the current position, we can either consider the actuator has infinite authority or
        it is limited)
        """
        assert type(command) == self.valid_command, f"Input command must be an instance of {self.valid_command} but is an {type(command)} object"
        return self.__dynamics__(command, *args, v_r=v_r, **kwargs)

    @abstractmethod
    def __dynamics__(self, command:Command, *args, v_r:tuple=None, **kwargs) -> GeneralizedForces:
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
    def f_min(self) -> float:
        return self._f_min
    
    @property
    def f_max(self) -> float:
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
    
    @property
    def dt(self) -> float:
        return self._dt
    
    @dt.setter
    def dt(self, value:float) -> None:
        self._dt = value

    @property
    def valid_command(self):
        return Command

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
                 dt:float,
                 *args,
                 f_min:float=-float('inf'),
                 f_max:float=float('inf'),
                 angle_0:float=0,
                 **kwargs):
        super().__init__(xy, angle, u_min, u_max, f_min, f_max, dt, *args, **kwargs)
        self._angle = angle_0

    def __dynamics__(self, command:Command, *args, v_r:tuple=None, **kwargs) -> GeneralizedForces:
        """
        command: (Rudder's angle)
        """
        return GeneralizedForces()
    
    @property
    def valid_command(self):
        return Command
    
class ControlSurface(Actuator):
    """
    Control surfaces: Control surfaces can be mounted at different locations to produce lift and drag forces.
    """
    def __init__(self,
                 xy:tuple,
                 angle:float,
                 u_min:tuple,
                 u_max:tuple,
                 dt:float,
                 *args,
                 f_min:float=-float('inf'),
                 f_max:float=float('inf'),
                 angle_0:float=0,
                 **kwargs):
        super().__init__(xy, angle, u_min, u_max, f_min, f_max, dt, *args, **kwargs)
        self._angle = angle_0

    def __dynamics__(self, command:Command, *args, v_r:tuple=None, **kwargs) -> GeneralizedForces:
        """
        command: (control surface's angle)
        """
        return GeneralizedForces()
    
    @property
    def valid_command(self):
        return Command
    
class WaterJet(Actuator):
    """
    Water jets: Water jets are an alternative to main propellers aft of the ship. They are usually used for high-speed craft.
    """
    def __init__(self,
                 xy:tuple,
                 angle:float,
                 u_min:tuple,
                 u_max:tuple,
                 dt:float,
                 *args,
                 f_min:float=-float('inf'),
                 f_max:float=float('inf'),
                 angle_0:float=0,
                 **kwargs):
        super().__init__(xy, angle, u_min, u_max, f_min, f_max, dt, *args, **kwargs)
        self._angle = angle_0

    def __dynamics__(self, command:Command, *args, v_r:tuple=None, **kwargs) -> GeneralizedForces:
        """
        command: (control surface's angle)
        """
        return GeneralizedForces()
    
    @property
    def valid_command(self):
        return Command
    

class AzimuthThruster(Actuator):
    """
    any tuple containing information regarding a command is always made of (angle, speed)
    """
    def __init__(self,
                 xy:tuple,
                 angle:float, # Orientation when thruster is at its reference position
                 alpha_range:tuple,
                 v_range:tuple,
                 dt:float,
                 *args,
                 f_min:float=-float('inf'),
                 f_max:float=float('inf'),
                 alpha_0:float=0.0, # Orientation w.r.t. initial orientation -> alpha at t=0
                 speed_0:float=0.0, # Initial speed, i.e. speed at t=0
                 alpha_rate_max:float=float('inf'),
                 v_rate_max:float=float('inf'),
                 c_t:float=2.2,
                 **kwargs):
        super().__init__(
            xy,
            angle,
            (alpha_range[0], v_range[0]),
            (alpha_range[1], v_range[1]),
            f_min,
            f_max,
            dt,
            *args,
            u_rate_min=(-alpha_rate_max, -v_rate_max),
            u_rate_max=(alpha_rate_max, v_rate_max),
            **kwargs
        )
        self._alpha: float = alpha_0 # We don't add it to angle here, this value must be interpreted as "angle w.r.t initial angle"
        self._speed: float = speed_0
        self._ct: float = c_t # Force vs speed coefficient

    def __dynamics__(self, command:AzimuthThrusterCommand, *args, **kwargs) -> GeneralizedForces:
        """
        command: (angle in degrees, speed) --> The input command are always assuming 0 is aligned with heading
        """
        assert type(command) == self.valid_command, f"Input command must be an instance of {self.valid_command} but is an {type(command)} object"

        # Computing new angle
        actual_angle_in_ship_frame = self._alpha + self.angle_deg
        desired_rate_alpha = (command.angle - actual_angle_in_ship_frame)/self._dt # Compute desired angle rate
        rate_alpha_clipped = clip(desired_rate_alpha, self._u_rate_min[0], self._u_rate_max[0]) # Clip angle rate to satisfy constraints
        alpha_deg = self._alpha + rate_alpha_clipped * self._dt # Compute new angle
        self._alpha = clip(alpha_deg, self._u_min[0], self._u_max[0]) # Clip angle to satisfy constraints

        # Computing new speed
        desired_acc = (command.speed - self._speed)/self._dt # Compute desired acceleration (speed rate)
        acc_clipped = clip(desired_acc, self._u_rate_min[1], self._u_rate_max[1]) # Clip acceleration to satisfy constraints
        speed = self._speed + acc_clipped * self._dt # Compute new speed
        self._speed = clip(speed, self._u_min[1], self._u_max[1]) # Clip speed to satisfy constraints

        # Compute resulting force
        Ftot = self._ct * self._speed**2 # Compute total force
        Ftot_clipped = clip(Ftot, self._f_min, self._f_max) # Clip force to satisfy constraints -> [f_min, f_max]

        # Project force in x, y and torque
        tot_angle = (self._alpha + self.angle_deg)*pi/180.0
        Fx = Ftot_clipped * cos(tot_angle)
        Fy = -Ftot_clipped * sin(tot_angle)
        Nz = Fx * self._xy[1] - Fy * self._xy[0]
        return GeneralizedForces(f_x=Fx, f_y=Fy, tau_z=Nz)
    
    @property
    def alpha_min(self) -> float:
        return self._u_min[0]
    
    @property
    def alpha_max(self) -> float:
        return self._u_max[0]
    
    @property
    def v_min(self) -> float:
        return self._u_min[1]
    
    @property
    def v_max(self) -> float:
        return self._u_max[1]
    
    @property
    def valid_command(self):
        return AzimuthThrusterCommand
    
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
                 v_min:float,
                 v_max:float,
                 dt:float,
                 *args,
                 f_min:GeneralizedForces=None,
                 f_max:GeneralizedForces=None,
                 speed_0:float=0.0,
                 v_rate_max:float=float('inf'),
                 c_t:float=2.2,
                 **kwargs):
        super().__init__(xy, angle, (0, 0), (v_min, v_max), dt, *args, f_min=f_min, f_max=f_max, speed_0=speed_0, alpha_0=0, angle_rate_max=0, v_rate_max=v_rate_max, c_t=c_t, **kwargs)

    def __dynamics__(self, command:ThrusterCommand, *args, **kwargs) -> GeneralizedForces:
        """
        command: (propeller's speed)
        """
        return super().__dynamics__(command, *args, **kwargs)
    
    @property
    def valid_command(self):
        return ThrusterCommand
    
def clip(val:float, min_val:float, max_val:float) -> float:
    return min(max(val, min_val), max_val)

def test() -> None:
    from nav_env.ships.ship import Ship
    from nav_env.actuators.actuators import AzimuthThruster, Thruster, Rudder
    
    dt = 1.0
    a1 = AzimuthThruster((-20, 10), 0, (-360, 360), (-3,3), dt, alpha_rate_max=float('inf'), v_rate_max=float('inf'))
    a2 = Thruster((1., -1.), -5, (1.), (3.), dt)
    a3 = Rudder((-3, 0.), 0, -10, 10, dt, angle_0=-3)
    print(a1, a2, a3)
    print(a1.dynamics(AzimuthThrusterCommand(-90, 2)))
    print(a1._alpha, a1._speed)
    
    ship = Ship(actuators=[a1, a2, a3])
    print(ship)

if __name__ == "__main__":
    test()

