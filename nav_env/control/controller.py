from abc import ABC, abstractmethod
from nav_env.ships.states import States
from nav_env.control.command import GeneralizedForces, Command


class ControllerBase(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, states:States, *args, **kwargs) -> Command:
        pass

class Controller(ControllerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, states:States) -> GeneralizedForces:
        return GeneralizedForces(6 * [0.])