from nav_env.control.command import GeneralizedForces, Command
from nav_env.actuators.collection import ActuatorCollection

class ControlAllocationBase:
    """
    Input is a desired force, output is a list of actuator commands
    
    """
    def __init__(self, actuators:ActuatorCollection, *args, **kwargs):
        self._actuators = actuators
        

        # Setup the control allocation problem in any ways

    def get(self, force:GeneralizedForces, *args, **kwargs) -> list:
        commands = []
        for a in self._actuators:
            command = a.valid_command()
            commands.append(command)
        return commands