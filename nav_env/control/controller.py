from abc import ABC, abstractmethod
from nav_env.ships.states import States
from nav_env.control.command import GeneralizedForces, Command
from nav_env.actuators.collection import ActuatorCollection


class ControllerBase(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, states:States, desired_states:States, *args, **kwargs) -> Command:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass



class Controller(ControllerBase):
    def __init__(self, *args, actuators:ActuatorCollection=ActuatorCollection.empty(), **kwargs):
        """
        Controllers must eventually be aware of what are the available actuators -> Hence they can return list of commands that suit the real system. 
        The actuator's information can also be used for setting up optimal control formulation
        """
        self._actuators = actuators
        super().__init__(*args, **kwargs)

    def get(self, states:States, desired_states:States, *args, **kwargs) -> GeneralizedForces:
        return GeneralizedForces()
    
    def reset(self) -> None:
        pass

    @property
    def actuators(self) -> ActuatorCollection:
        return self._actuators
    
    @actuators.setter
    def actuators(self, value:ActuatorCollection) -> None:
        self._actuators = value

# class HeadingAndSpeedController(ControllerBase):
#     def __init__(self, )