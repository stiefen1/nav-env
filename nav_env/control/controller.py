from abc import ABC, abstractmethod
from nav_env.ships.states import States
from nav_env.control.command import GeneralizedForces, Command
from nav_env.actuators.collection import ActuatorCollection
from nav_env.control.allocation import ControlAllocation
from typing import Union


class ControllerBase(ABC):
    def __init__(self, actuators:ActuatorCollection, allocation:ControlAllocation, *args, **kwargs):
        self._actuators = actuators
        self._allocation = allocation

    def get(self, states:States, desired_states:States, *args, **kwargs) -> Union[list[Command], GeneralizedForces]:
        return self.allocation.get((self.__get__(states, desired_states, *args, **kwargs)))

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