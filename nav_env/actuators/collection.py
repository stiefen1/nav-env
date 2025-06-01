from nav_env.actuators.actuators import Actuator, Rudder, Thruster, AzimuthThruster
from nav_env.control.command import GeneralizedForces, Command
from math import pi

class ActuatorCollection:
    def __init__(self, actuators:list[Actuator], *args, **kwargs):
        self._actuators = actuators

    def append(self, actuator:Actuator) -> None:
        assert isinstance(actuator, Actuator), f"Obstacle must be an instance of Actuator not {type(actuator)}"
        return self._actuators.append(actuator)

    def remove(self, actuator:Actuator) -> None:
        return self._actuators.remove(actuator)
    
    def dynamics(self, commands:list[Command], *args, v_r:tuple=None, **kwargs) -> GeneralizedForces:
        sum_of_forces = GeneralizedForces()
        for command, actuator in commands, self._actuators:
            sum_of_forces += actuator.dynamics(command, *args, v_r=v_r, **kwargs)
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

    # def __getattr__(self, name):
    #     list_of_attributes = []
    #     for a in self._actuators:
    #         list_of_attributes.append(a.__getattribute__(name))
    #     return list_of_attributes
    
    @staticmethod
    def empty() -> "ActuatorCollection":
        return ActuatorCollection([])
    
    @property
    def rudders(self) -> "ActuatorCollection":
        return ActuatorCollection(get_all_objects_of_type_in_iterable(self, Rudder))
    
    @property
    def thrusters(self) -> "ActuatorCollection":
        return ActuatorCollection(get_all_objects_of_type_in_iterable(self, Thruster))
    
    @property
    def azimuth_thrusters(self) -> "ActuatorCollection":
        return ActuatorCollection(get_all_objects_of_type_in_iterable(self, AzimuthThruster))
    
def get_all_objects_of_type_in_iterable(list_of_obj, obj_type) -> list:
    out = []
    for obj in list_of_obj:
        if type(obj) == obj_type:
            out.append(obj)
    return out

def test() -> None:
    from nav_env.actuators.actuators import AzimuthThruster, Thruster, Rudder
    a1 = AzimuthThruster((0., 0.), 10., (0., 0.), (1., 1.), 1.)
    a2 = Thruster((0., 0.), 10., (0.), (1.), 1.)
    a3 = Rudder((0., 0.), 10., 1, 3, 1.)
    a4 = AzimuthThruster((0., 0.), -10, (-10, 10), (-20, 20), 1.)
    c = ActuatorCollection([a1, a2, a3, a4])
    print(c)
    print(c.rudders)
    print(c.thrusters)
    print(c.azimuth_thrusters)

if __name__ == "__main__":
    test()