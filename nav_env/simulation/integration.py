from abc import ABC, abstractmethod
from nav_env.control.states import States, TimeDerivatives, DeltaStates

DEFAULT_INTEGRATION_DT = 0.1

class Integrator:
    def __init__(self, dt: float):
        self.dt = dt

    def __call__(self, states: States, derivative: TimeDerivatives) -> tuple[States, DeltaStates]:
        return self.__integrate__(states, derivative)

    #TODO: Replace TimeDerivatives by something more complex that 
    # allows to gather multiple differentials
    @abstractmethod
    def __integrate__(self, states: States, derivative: TimeDerivatives) -> tuple[States, DeltaStates]:
        raise NotImplementedError
    
class Euler(Integrator):
    def __init__(self, dt: float = DEFAULT_INTEGRATION_DT):
        super().__init__(dt)

    def __integrate__(self, states: States, derivative: TimeDerivatives) -> tuple[States, DeltaStates]:
        dx = derivative * self.dt
        return states + dx, dx # We return dx as well to update enveloppe easily