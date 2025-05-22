from nav_env.actuators.actuators import Actuator
from nav_env.control.command import GeneralizedForces
from math import pi

class ActuatorCollection(Actuator):
    def __init__(self, actuators:list[Actuator], *args, **kwargs):
        self._actuators = actuators

    def append(self, actuator:Actuator) -> None:
        assert isinstance(actuator, Actuator), f"Obstacle must be an instance of Actuator not {type(actuator)}"
        return self._actuators.append(actuator)

    def remove(self, actuator:Actuator) -> None:
        return self._actuators.remove(actuator)
    
    def dynamics(self, commands:list[tuple], vr:tuple) -> GeneralizedForces:
        sum_of_forces = GeneralizedForces()
        for command, actuator in commands, self._actuators:
            sum_of_forces += actuator.dynamics(command, vr)
        return sum_of_forces
    
    def __getitem__(self, index: int) -> Actuator:
        return self._actuators[index]
    
    def __len__(self) -> int:
        return len(self._actuators)

    def __repr__(self):
        return f"ActuatorCollection({len(self._actuators)} actuators: {[a for a in self._actuators]})"

    def __iter__(self):
        for obs in self._actuators:
            yield obs

    def __getattr__(self, name):
        list_of_attributes = []
        for a in self._actuators:
            list_of_attributes.append(a.__getattribute__(name))
        return list_of_attributes

def test() -> None:
    from nav_env.actuators.actuators import AzimuthThruster, Thruster, Rudder
    a1 = AzimuthThruster((0., 0.), 10., (0., 0.), (1., 1.), GeneralizedForces(), GeneralizedForces())
    a2 = Thruster((0., 0.), 10., (0.), (1.), GeneralizedForces(), GeneralizedForces())
    a3 = Rudder((0., 0.), 10., 1, 3, GeneralizedForces(), GeneralizedForces())
    c = ActuatorCollection([a1, a2, a3])
    print(c.u_max)

if __name__ == "__main__":
    test()