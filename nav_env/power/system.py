"""
When the control system asks for a command, it must go through a PowerSystem object
that assess whether there is enough power for it. If not, the command effectively applied
to the ship are changed accordingly.

"""

from typing import Any
from nav_env.control.command import Command
from abc import ABC, abstractmethod

class PowerSystemBase(ABC):
    def __init__(self, system:Any, max_power_available:float, *args, **kwargs):
        self.system = system
        self.max_power_available = max_power_available

    def power_consumption(commands:list[Command]) -> float:
        power_cons = 0.0
        for command in commands:
            power_cons += command.power()
        return power_cons

    def get(self, commands:list[Command]) -> list[Command]:
        required_power = self.power_consumption(commands)
        if  required_power <= self.max_power_available:
            return commands
        else:
            return self.distribute_all_power(commands)
        
    @abstractmethod
    def distribute_all_power(self, commands:list[Command]) -> list[Command]:
        pass

class PowerSystem(PowerSystemBase):
    def __init__(self, system:Any, max_power_available:float=10.0, *args, **kwargs):
        super().__init__(system, max_power_available, *args, **kwargs)

    def distribute_all_power(self, commands:list[Command]) -> list[Command]:
        return commands

