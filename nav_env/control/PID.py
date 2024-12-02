from nav_env.control.controller import Controller
from nav_env.ships.states_old import ShipStates
from nav_env.control.command import GeneralizedForces

class PID(Controller):
    def __init__(self, kp, ki, kd):
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral = 0

    def get(self, states:ShipStates) -> GeneralizedForces:
        pass