from abc import ABC, abstractmethod
from nav_env.control.command import GeneralizedForces, AzimuthThrusterCommand, ThrusterCommand, Command, AzimuthThrusterSpeedCommand
from math import pi, cos, sin
from typing import Union
import casadi as cd
import numpy as np
import warnings

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
        self._logs = np.zeros((0, self.nu))
        self._last_command = None

    def save(self) -> None:
        if self._last_command is not None:
            self._logs = np.append(self._logs, self._last_command.to_numpy().reshape(1, self.nu), axis=0)
        else:
            warnings.warn(f"Last command of {self} is None - save aborted")

    def sample_within_bounds(self, as_list:bool=False) -> list:
        """Sample a random control command within bounds"""
        sample_as_list =np.random.uniform(self._u_min, self._u_max).tolist()
        return sample_as_list if as_list else self.valid_command(*sample_as_list) 

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
        self._last_command = command
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
    
    def u_at_min_power(self) -> tuple:
        return tuple([0.0] * len(self._u_min))
        
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
    def u_rate_max(self) -> tuple:
        return self._u_rate_max
    
    @property
    def u_rate_min(self) -> tuple:
        return self._u_rate_min
    
    @property
    def u_mean(self) -> tuple:
        return tuple([0.5*(ui_max + ui_min) for (ui_min, ui_max) in zip(self._u_min, self._u_max)])
    
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
                 alpha_0:float=0.0, # Orientation w.r.t. initial orientation -> alpha at t=0 in degrees
                 speed_0:float=0.0, # Initial speed, i.e. speed at t=0
                 alpha_rate_max:float=float('inf'),
                 v_rate_max:float=float('inf'),
                 c_t:float=2.2, # *60,
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
    
    def __dynamics__(self, command:Union[AzimuthThrusterCommand,tuple], *args, do_clip:bool=True, use_casadi:bool=False, output_type:str='force', **kwargs) -> GeneralizedForces:
        """
        command: (angle in degrees, speed) --> The input command are always assuming 0 is aligned with heading

        do_clip, output_type, tuple for command are just here to be able to integrate this actuator as part of an NMPC formulation

        The complexity of this method is mainly due to its potential use with casadi
        """
        assert type(command) in [self.valid_command, tuple], f"Input command must be an instance of {self.valid_command} or a tuple but is an {type(command)} object"

        # If use_casadi flag is True, disable clip as constraints must be included in the optimization formulation and casadi's min max would need some refactoring to be integrated as cos,sin
        do_clip = False if use_casadi else do_clip

        if type(command) == self.valid_command:
            angle = command.angle
            speed = command.speed
        else:
            angle = command[0]
            speed = command[1]

        if use_casadi:
            self.cos = cd.cos
            self.sin = cd.sin
        else:
            self.cos = cos
            self.sin = sin

        # Computing new angle
        rate_alpha = self.get_rate_alpha(angle, do_clip=do_clip)
        self._alpha = self.get_alpha(rate_alpha, do_clip=do_clip)

        # Computing new speed
        acc = self.get_acc(speed, do_clip=do_clip)
        self._speed = self.get_speed(acc, do_clip=do_clip)

        # Compute resulting force
        Ftot = self.get_Ftot(do_clip=do_clip)

        # Project force in x, y and torque
        tot_angle = (self._alpha + self.angle_deg)*pi/180.0
        
        # Project total force
        Fx = Ftot * self.cos(tot_angle)
        Fy = -Ftot * self.sin(tot_angle)
        Nz = -Fx * self._xy[1] + Fy * self._xy[0]
        force = GeneralizedForces(f_x=Fx, f_y=Fy, tau_z=Nz)
        return force if output_type == 'force' else force.to_numpy()
    
    def u_at_min_power(self) -> tuple:
        return (None, 0.0)
    
    def get_rate_alpha(self, alpha:float, do_clip:bool=True) -> float:
        actual_angle_in_ship_frame = self._alpha + self.angle_deg
        rate_alpha = (alpha - actual_angle_in_ship_frame)/self._dt # Compute desired angle rate
        if do_clip:
            rate_alpha = clip(rate_alpha, self._u_rate_min[0], self._u_rate_max[0]) # Clip angle rate to satisfy constraints
        return rate_alpha
    
    def get_alpha(self, rate_alpha:float, do_clip:bool=True) -> float:
        alpha_deg = self._alpha + rate_alpha * self._dt # Compute new angle
        if do_clip:
            alpha_deg = clip(alpha_deg, self._u_min[0], self._u_max[0]) # Clip angle to satisfy constraints
        return alpha_deg
    
    def get_acc(self, speed:float, do_clip:bool=True) -> float:
        acc = (speed - self._speed)/self._dt # Compute desired acceleration (speed rate)
        if do_clip:
            acc = clip(acc, self._u_rate_min[1], self._u_rate_max[1]) # Clip acceleration to satisfy constraints
        return acc
    
    def get_speed(self, acc:float, do_clip:bool=True) -> float:
        speed = self._speed + acc * self._dt # Compute new speed
        if do_clip:
            speed = clip(speed, self._u_min[1], self._u_max[1]) # Clip speed to satisfy constraints
        return speed
    
    def get_Ftot(self, do_clip:bool=True) -> float:
        Ftot = self._ct * self._speed**2 # Compute total force
        if do_clip:
            Ftot = clip(Ftot, self._f_min, self._f_max) # Clip force to satisfy constraints -> [f_min, f_max]
        return Ftot
    

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
    
class AzimuthThrusterWithSpeed(Actuator):
    """
    any tuple containing information regarding a command is always made of (angle, speed)
    """
    def __init__(self,
                 xy:tuple,
                 angle:float, # Orientation when thruster is at its reference position
                 azimuth_rate_range:tuple, # RADIANS / SECONDS
                 v_range:tuple,
                 dt:float,
                 *args,
                 f_min:float=-float('inf'),
                 f_max:float=float('inf'),
                 alpha_0:float=0.0, # Orientation w.r.t. initial orientation -> alpha at t=0
                 azimuth_rate_0:float=0.0, # azimuth rate at t=0
                 speed_0:float=0.0, # Initial speed, i.e. speed at t=0
                 azimuth_acc_max:float=float('inf'),
                 v_rate_max:float=float('inf'),
                 c_t:float=2.2, # *60,
                 **kwargs):
        super().__init__(
            xy,
            angle,
            (azimuth_rate_range[0], v_range[0]),
            (azimuth_rate_range[1], v_range[1]),
            f_min,
            f_max,
            dt,
            *args,
            u_rate_min=(-azimuth_acc_max, -v_rate_max),
            u_rate_max=(azimuth_acc_max, v_rate_max),
            **kwargs
        )
        self._alpha: float = alpha_0 # We don't add it to angle here, this value must be interpreted as "angle w.r.t initial angle"
        self._azimuth_rate: float = azimuth_rate_0 # AZIMUTH RATE IS IN RADIANS / SEC
        self._speed: float = speed_0
        self._ct: float = c_t # Force vs speed coefficient
    
    def __dynamics__(self, command:Union[AzimuthThrusterSpeedCommand,tuple], *args, do_clip:bool=True, use_casadi:bool=False, output_type:str='force', **kwargs) -> GeneralizedForces:
        """
        command: (angle in degrees, speed) --> The input command are always assuming 0 is aligned with heading

        do_clip, output_type, tuple for command are just here to be able to integrate this actuator as part of an NMPC formulation

        The complexity of this method is mainly due to its potential use with casadi
        """
        assert type(command) in [self.valid_command, tuple], f"Input command must be an instance of {self.valid_command} or a tuple but is an {type(command)} object"

        # If use_casadi flag is True, disable clip as constraints must be included in the optimization formulation and casadi's min max would need some refactoring to be integrated as cos,sin
        do_clip = False if use_casadi else do_clip

        if type(command) == self.valid_command:
            # print(command)
            desired_azimuth_rate = command.azimuth_rate
            propeller_speed = command.propeller_speed
        else:
            desired_azimuth_rate = command[0]
            propeller_speed = command[1]

        if use_casadi:
            self.cos = cd.cos
            self.sin = cd.sin
        else:
            self.cos = cos
            self.sin = sin

        # Computing new angle
        # rate_alpha = self.get_rate_alpha(angle, do_clip=do_clip)
        # self._alpha = self.get_alpha(rate_alpha, do_clip=do_clip)
        # print("0: ", float(self._alpha), float(self._azimuth_rate), float(pi), float(self._dt))
        self._alpha += self._azimuth_rate * 180/pi * self._dt # Integrate alpha
        azimuth_acc = self.get_azimuth_acc(desired_azimuth_rate, do_clip=do_clip)
        self._azimuth_rate = self.get_azimuth_rate(azimuth_acc, do_clip=do_clip) # Integrate azimuth acceleration
        # print("rate+alpha: ", float(self._azimuth_rate), float(self._alpha), float(self._angle_deg))
        
        # Computing new speed
        propeller_acc = self.get_acc(propeller_speed, do_clip=do_clip)
        self._speed = self.get_speed(propeller_acc, do_clip=do_clip)
        # print(self._speed)

        # Compute resulting force
        Ftot = self.get_Ftot(do_clip=do_clip)
        # print("Ftot: ", Ftot)

        # Project force in x, y and torque
        tot_angle_rad = (self._alpha + self.angle_deg)*pi/180.0
        
        # Project total force
        Fx = Ftot * self.cos(tot_angle_rad)
        Fy = -Ftot * self.sin(tot_angle_rad)
        Nz = -Fx * self._xy[1] + Fy * self._xy[0]
        force = GeneralizedForces(f_x=Fx, f_y=Fy, tau_z=Nz)
        
        return force if output_type == 'force' else force.to_numpy()
    
    def get_azimuth_acc(self, desired_azimuth_rate:float, do_clip:bool=True) -> float:
        # print("1.1: ", float(desired_azimuth_rate), float(self._azimuth_rate), self._dt)
        desired_azimuth_acc = (desired_azimuth_rate - self._azimuth_rate)/self._dt # Compute desired azimuth acceleration
        if do_clip:
            azimuth_acc = clip(desired_azimuth_acc, self._u_rate_min[0], self._u_rate_max[0]) # Clip azimuth acceleration to satisfy constraints
            return azimuth_acc
        else:
            return desired_azimuth_acc
        
    def get_azimuth_rate(self, azimuth_acc:float, do_clip:bool=True) -> float:
        # print("2.1: ", float(azimuth_acc), do_clip)
        azimuth_rate = self._azimuth_rate + azimuth_acc * self._dt
        if do_clip:
            azimuth_rate = clip(azimuth_rate, self._u_min[0], self._u_max[0])
        # print("2.2: ", azimuth_rate, self._u_min[0], self._u_max[0])
        return azimuth_rate

    
    def get_acc(self, speed:float, do_clip:bool=True) -> float:
        acc = (speed - self._speed)/self._dt # Compute desired acceleration (speed rate)
        if do_clip:
            acc = clip(acc, self._u_rate_min[1], self._u_rate_max[1]) # Clip acceleration to satisfy constraints
        return acc
    
    def get_speed(self, acc:float, do_clip:bool=True) -> float:
        speed = self._speed + acc * self._dt # Compute new speed
        if do_clip:
            speed = clip(speed, self._u_min[1], self._u_max[1]) # Clip speed to satisfy constraints
        return speed
    
    def get_Ftot(self, do_clip:bool=True) -> float:
        Ftot = self._ct * self._speed**2 # Compute total force
        if do_clip:
            Ftot = clip(Ftot, self._f_min, self._f_max) # Clip force to satisfy constraints -> [f_min, f_max]
        return Ftot

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
        return AzimuthThrusterSpeedCommand
    
    
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

