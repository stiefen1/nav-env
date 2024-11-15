from abc import ABC, abstractmethod
from nav_env.ships.states import ShipStates
from nav_env.environment import NavigationEnvironment
from nav_env.geometry.vector import Vector
from nav_env.ships.params import ShipPhysicalParams

DEFAULT_SHIP_GEOMETRY = [[
    (0, 0),
    (0, 1),
    (1, 1),
    (1, 0),
    (0, 0)
]]

class ShipBase(ABC):
    def __init__(self, states:ShipStates, physical_params:ShipPhysicalParams, geometry=DEFAULT_SHIP_GEOMETRY):
        self._states = states
        self._geometry = geometry # Maybe geometry should be part of the params? But params is more for physical properties.
        self._physical_params = physical_params

    def draw(self):
        """
        Draw the ship for pygame.
        """
        pass

    def plot(self, ax=None, **kwargs):
        """
        Plot the ship.
        """
        pass

    @abstractmethod
    def integrate(self, dt:float):
        """
        Integrate the ship states.
        """
        pass

    @abstractmethod
    def update_derivatives(self, x, y, psi, u, v, r, delta):
        """
        Update the derivatives of the states.
        """
        pass

class SimpleShip(ShipBase):
    def __init__(self, states:ShipStates):
        super().__init__(states)

    def integrate(self, dt:float):
        """
        Integrate the ship states.
        """
        self._states += self._states * dt

    def update_derivatives(self, x, y, psi, u, v, r, delta):
        """
        Update the derivatives of the states.
        """
        self._states.velocity = ...
        self._states.yaw_rate = ...
        self._states.acceleration = ...
        self._states.yaw_acceleration = ...

    @property
    def states(self) -> ShipStates:
        return self._states

class Ship(ShipBase):
    def __init__(self, states:ShipStates, environment:NavigationEnvironment):
        self._states = states
        self._env = environment

    def integrate(self, dt:float):
        """
        Integrate the ship states.
        """
        pass

    def update_derivatives(self, x, y, psi, u, v, r, delta):
        """
        Update the derivatives of the states.
        """
        self._states.velocity = ...
        self._states.yaw_rate = ...
        self._states.acceleration = ...
        self._states.yaw_acceleration = ...

    @staticmethod
    def get_generalized_force_from_wind(x, y, psi, u, v, r, delta, wind:Vector) -> Vector:
        """
        Get the generalized forces acting on the ship.
        """
        pass

    @staticmethod
    def get_generalized_force_from_water(x, y, psi, u, v, r, delta, water:Vector) -> Vector:
        """
        Get the generalized forces acting on the ship.
        """
        pass

    @staticmethod
    def get_generalized_force_from_env(x, y, psi, u, v, r, delta, env:NavigationEnvironment) -> Vector:
        """
        Get the generalized forces acting on the ship.
        """
        wind_force:Vector = Ship.get_generalized_force_from_wind(x, y, psi, u, v, r, delta, env.wind_source(x, y))
        water_force:Vector = Ship.get_generalized_force_from_water(x, y, psi, u, v, r, delta, env.water_source(x, y))
        return wind_force + water_force

    @staticmethod
    def get_generalized_control_forces(delta, u, v, r, env:NavigationEnvironment) -> Vector:
        """
        Get the generalized forces due to control inputs.
        """
        pass

    @property
    def states(self) -> ShipStates:
        return self._states
    
    @property
    def environment(self) -> NavigationEnvironment:
        return self._env
    