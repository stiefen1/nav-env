from nav_env.control.controller import ControllerBase
from nav_env.ships.states import States
from nav_env.control.command import GeneralizedForces

# TODO: Implement a keyboard controller that allows to control the ship with the keyboard

class KeyboardController(ControllerBase):
    def __init__(self):
        pass

    def get(self, states:States) -> GeneralizedForces:
        return GeneralizedForces(*(states.dim * [0.]))

def test():
    a = KeyboardController()
    s0 = States(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    f1 = a.get(s0)
    print(f1)


if __name__ == "__main__":
    test()