from abc import ABC, abstractmethod
from dataclasses import dataclass

# class Fault ?

"""
Not necessarily useful classes, but more to normalize the interface

Fault can happen in three different ways:
- randomly, using probability distribution
- after a given amount of time
- manually, using the "activate" method

How to specify these different behaviors is not clear yet 
"""


# @dataclass
# class ProbabilityParameters:
#     distribution:str = 'gauss'

# @dataclass
# class TimeParameters:



class Fault(ABC):
    def __init__(self, active:bool, mode:dict, *args, **kwargs):
        self.mode = mode
        self.active = active

    def activate(self, *args, **kwargs) -> None:
        self.active = True

    def sample(self) -> None:
        if self.active:
            return
        
        # TODO: Make fault active depending on probability distribution


class AdditiveFault(Fault):
    def __init__(self, *args, active:bool=False, mode:dict={}, **kwargs):
        super().__init__(*args, active=active, mode=mode, **kwargs)

class MultiplicativeFault(Fault):
    def __init__(self, *args, active:bool=False, mode:dict={}, **kwargs):
        super().__init__(*args, active=active, mode=mode, **kwargs)

class FaultsCollection(list):
    def __init__(self, faults:list[Fault]=None, *args, **kwargs):
        self._faults = faults or []

    @property
    def active(self) -> tuple[bool]:
        return tuple([f.active for f in self._faults])


# Example:
class LossOfEffectiveness(MultiplicativeFault):
    def __init__(self, *args, active:bool=False, mode:dict={}, **kwargs):
        super().__init__(*args, active=active, mode=mode, **kwargs)