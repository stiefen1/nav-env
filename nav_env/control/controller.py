from abc import ABC, abstractmethod
from nav_env.ships.states import States
from nav_env.control.command import GeneralizedForces, Command
from nav_env.actuators.collection import ActuatorCollection
from nav_env.control.allocation import ControlAllocation
from nav_env.ships.physics import ShipPhysics
from typing import Union


class ControllerBase(ABC):
    def __init__(self, actuators:ActuatorCollection, allocation:ControlAllocation, *args, ship_physics:ShipPhysics=None, wind_feedforward:bool=False, **kwargs):
        self._actuators = actuators
        self._allocation = allocation
        if wind_feedforward:
            assert ship_physics is not None, f"ship_physics must be specified for adding wind feedforward term"
        self._ship_physics = ship_physics
        self._wind_feedforward = wind_feedforward

    def get(self, states:States, desired_states:States, *args, wind=None, **kwargs) -> Union[list[Command], GeneralizedForces]:
        forces_or_commands = self.__get__(states, desired_states, *args, **kwargs)
        if isinstance(forces_or_commands, GeneralizedForces) and self._wind_feedforward:
            forces_or_commands -= self._ship_physics.get_wind_force(wind, *states.xy_dot, states.psi_rad)
        return self.allocation.get((forces_or_commands), *args, **kwargs)

    @abstractmethod
    def __get__(self, states:States, desired_states:States, *args, **kwargs) -> Union[list[Command], GeneralizedForces]:
        return GeneralizedForces()

    @abstractmethod
    def reset(self) -> None:
        pass

    @property
    def actuators(self) -> ActuatorCollection:
        return self._actuators
    
    @actuators.setter
    def actuators(self, value:ActuatorCollection) -> None:
        self._actuators = value

    @property
    def allocation(self) -> ControlAllocation:
        return self._allocation
    
    @allocation.setter
    def allocation(self, value:ControlAllocation) -> None:
        self._allocation = value

class Controller(ControllerBase):
    def __init__(self, *args, actuators:Union[list, ActuatorCollection]=ActuatorCollection.empty(), allocation:ControlAllocation=ControlAllocation.empty(), **kwargs):
        """
        Controllers must eventually be aware of what are the available actuators -> Hence they can return list of commands that suit the real system. 
        The actuator's information can also be used for setting up optimal control formulation
        """
        actuators = actuators if isinstance(actuators, ActuatorCollection) else ActuatorCollection(actuators)
        super().__init__(actuators, allocation, *args, **kwargs)
    
    def __get__(self, states:States, desired_states:States, *args, **kwargs) -> Union[list[Command], GeneralizedForces]:
        return GeneralizedForces()
    
    def reset(self) -> None:
        pass

    

# class HeadingAndSpeedController(ControllerBase):
#     def __init__(self, )