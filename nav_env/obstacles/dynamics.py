from nav_env.obstacles.collection import ObstacleCollection
from nav_env.ships.states import States3, TimeDerivatives3, DeltaStates3
from nav_env.control.command import GeneralizedForces
from nav_env.simulation.integration import Integrator, Euler
from nav_env.obstacles.ship import EnveloppeBase, EmptyEnveloppe, ShipEnveloppe
# import nav_env.ships.physics as phy
from nav_env.ships.physics import PhysicsBase, ZeroAccelerationPhysics, ShipPhysics, ZeroAccelerationSailingShipPhysics
import pygame, matplotlib.pyplot as plt
from nav_env.environment.wind_vector import WindVector
from nav_env.environment.water_vector import WaterVector
from abc import abstractmethod, ABC
from typing import Callable

from copy import deepcopy


class BaseMovingObstacle(ABC):
    def __init__(self,
                 states:States3,
                 enveloppe:EnveloppeBase=None,
                 integrator:Integrator=None,
                 derivatives:TimeDerivatives3=None,
                 id:int=0,
                 ):
        self._states = states
        self._initial_states = deepcopy(states) # We don't want to share the address of the states
        self._integrator = integrator or Euler()
        self._derivatives = derivatives or TimeDerivatives3()
        self._enveloppe = enveloppe or EmptyEnveloppe() # Initialize enveloppe to an empty enveloppe
        self._id = id
        self._dx = None # Initialize differential to None
        self._accumulated_dx = DeltaStates3(0., 0., 0., 0., 0., 0.) # Initialize accumulated differential to 0
        self._generalized_forces = GeneralizedForces() # Initialize generalized forces acting on the ship to 0

    @abstractmethod
    def update_derivatives(self, wind:WindVector, water:WaterVector, control_forces:GeneralizedForces, external_forces:GeneralizedForces):
        """
        Update the derivatives of the states.
        """
        # MAYBE WE SHOULD ONLY CONSIDER GENERALIZED FORCE TO SIMPLIFY AT FIRST
        # self._generalized_forces, self._derivatives = self._physics.get_time_derivatives_and_forces(self._states, wind, water, control_forces, external_forces)
        pass

    @abstractmethod
    def rotation_matrix(self, angle:float, dim:int=2):
        """
        Get the rotation matrix.
        """
        pass

    # def get_initial_enveloppe(self):
    #     """
    #     Initialize the enveloppe.
    #     """
    #     return ShipEnveloppe(length=self._physics.length, width=self._physics.width).rotate_and_translate(self._states.x, self._states.y, self._states.psi_deg)

    def step(self, wind:WindVector, water:WaterVector, external_forces:GeneralizedForces=GeneralizedForces(), update_enveloppe=True):
        """
        Step the ship.
        """
        self.update_derivatives(wind, water, external_forces)
        self.integrate()

        # This is a trick, to control when to update the enveloppe
        # Since it is very time consuming, we can update it only when we need it
        # using self._accumulated_dx 
        if update_enveloppe:
            self.update_enveloppe()
        else:
            self._accumulated_dx += self._dx
            
    def integrate(self):
        """
        Integrate the ship states.
        """
        self._states, self._dx = self._integrator(self._states, self._derivatives)

    def reset(self):
        """
        Reset the ship to its initial state.
        """
        self._states = deepcopy(self._initial_states) # We don't want to share the address of the states
        self._enveloppe.reset()
        self._dx = None
        self._generalized_forces = GeneralizedForces()

    def update_enveloppe(self):
        if self._dx is not None:
            self._enveloppe.rotate_and_translate(self._dx[0], self._dx[1], self._dx[2])
        else:
            raise UserWarning(f"self._dx is None, you must first call integrate()")

    def get_constant_speed_fn(self) -> Callable[[float], States3]:
        """
        Get a constant speed function describing the ship's pose at a given time.
        """
        return lambda t: (
            States3(self._states.x + self._states.x_dot * t,
                    self._states.y + self._states.y_dot * t,
                    self._states.psi_deg + self._states.psi_dot_deg * t,
                    self._states.x_dot,
                    self._states.y_dot,
                    self._states.psi_dot_deg
            )
        )
    
    def get_constant_acceleration_fn(self) -> Callable[[float], States3]:
        """
        Get a constant acceleration function describing the ship's pose at a given time.
        """
        return lambda t: (
            States3(self._states.x + self._states.x_dot * t + 0.5 * self._derivatives.x_dot_dot * t**2,
                    self._states.y + self._states.y_dot * t + 0.5 * self._derivatives.y_dot_dot * t**2,
                    self._states.psi_deg + self._states.psi_dot_deg * t + 0.5 * self._derivatives.psi_dot_dot * t**2,
                    self._states.x_dot + self._derivatives.x_dot_dot * t,
                    self._states.y_dot + self._derivatives.y_dot_dot * t,
                    self._states.psi_dot_deg + self._derivatives.psi_dot_dot * t
            )
        )

    def draw(self, screen:pygame.Surface, *args, scale=1, keys=['enveloppe'], **kwargs):
        """
        Draw the ship for pygame.
        """
        if 'enveloppe' in keys:
            self._enveloppe.draw(screen, *args, scale=scale, color=(10, 10, 10), **kwargs)
        if 'frame' in keys:
            self.draw_frame(screen, *args, scale=scale, **kwargs)
        if 'velocity' in keys:
            self._states.draw(screen, *args, scale=scale, color=(255, 165, 0), **kwargs)
        if 'acceleration' in keys:
            self._derivatives.draw_acc(screen, self._states.xy, *args, scale=scale, color=(160, 32, 240), **kwargs)
        if 'forces' in keys:
            self._generalized_forces.draw(screen, self._states.xy, *args, scale=scale, unit_scale=1e-4, color=(0, 0, 0), **kwargs)

    def plot(self, keys=['enveloppe'], ax=None, **kwargs):
        """
        Plot the ship. Add 'enveloppe', 'frame', 'acceleration', 'velocity', 'forces' to keys to plot the corresponding elements.
        """
        # TODO: Add forces / acceleration / speed / frame of reference to the plot
        if ax is None:
            _, ax = plt.subplots()
        if 'enveloppe' in keys:
            self._enveloppe.plot(ax=ax, **kwargs)
        if 'frame' in keys:
            self.plot_frame(ax=ax, **kwargs)
        if 'acceleration' in keys:
            self._derivatives.plot_acc(self._states.xy, ax=ax, color='purple', angles='xy', scale_units='xy', scale=5e-3, **kwargs)
        if 'velocity' in keys:
            self._states.plot(ax=ax, color='orange', angles='xy', scale_units='xy', scale=1e-1,  **kwargs)
        if 'forces' in keys:
            self._generalized_forces.plot(self._states.xy, ax=ax, color='black', angles='xy', scale_units='xy', scale=1e3, **kwargs)

        # print(self.id, self._physics.generalized_forces.f_x / self._derivatives.x_dot_dot, self._physics.generalized_forces.f_y / self._derivatives.y_dot_dot)
        return ax
    
    def plot_frame(self, ax=None, **kwargs):
        """
        Plot the ship frame.
        """
        if ax is None:
            _, ax = plt.subplots()

        R = self.rotation_matrix(self._states.psi_rad, dim=2)
        ax.quiver(*self._states.xy, *R[0, :], color='r', **kwargs)
        ax.quiver(*self._states.xy, *R[1, :], color='g', **kwargs)
        return ax
    
    def draw_frame(self, screen:pygame.Surface, *args, scale=1, **kwargs):
        """
        Draw the ship frame for pygame.
        """
        R = self._physics.rotation_matrix(self._states.psi_rad, dim=2)
        screen_size = screen.get_size()
        x, y = self._states.x, self._states.y
        pygame.draw.line(screen, 
                         (255, 0, 0),
                         (scale*x + screen_size[0] // 2,
                         screen_size[1] // 2 - scale*y),
                         (scale*x + screen_size[0] // 2 + 100*scale*R[0, 0],
                          screen_size[1] // 2 - scale*y - 100*scale*R[1, 0]),
                          *args, **kwargs)
        pygame.draw.line(screen,
                         (0, 255, 0),
                         (scale*x + screen_size[0] // 2,
                          screen_size[1] // 2 - scale*y),
                          (scale*x + screen_size[0] // 2 + 100*scale*R[0, 1],
                           screen_size[1] // 2 - scale*y - 100*scale*R[1, 1]),
                           *args, **kwargs)

    def update_enveloppe_from_accumulation(self):
        """
        Update the enveloppe from the accumulated differential.

        WARNING!!! DONT USE THIS METHOD UNLESS YOU KNOW EXACTLY WHAT YOU ARE DOING
        """
        self._enveloppe.rotate_and_translate(self._accumulated_dx[0], self._accumulated_dx[1], self._accumulated_dx[2])
        self._accumulated_dx = DeltaStates3(0., 0., 0., 0., 0., 0.)

    def collide(self, obstacle:ObstacleCollection) -> bool:
        """
        Check if the ship collides with an obstacle.
        """
        for obs in obstacle:
            if self._enveloppe.intersects(obs):
                return True
        return False
    
    def __getattr__(self, name):
        return getattr(self._enveloppe, name)
    
    def __repr__(self):
        return f"{type(self).__name__}({self._states.__dict__})"

    
    @property
    def states(self) -> States3:
        return self._states
    
    @property
    def integrator(self) -> Integrator:
        return self._integrator
    
    @property
    def derivatives(self) -> TimeDerivatives3:
        return self._derivatives
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def enveloppe(self) -> EnveloppeBase:
        return self._enveloppe
    
    @property
    def position(self) -> tuple:
        return self._states.x, self._states.y
    
    # @property
    # def physics(self) -> PhysicsBase:
    #     return self._physics
    
    @property
    def generalized_forces(self) -> GeneralizedForces:
        """
        Get the generalized forces acting on the ship.
        """
        return self._generalized_forces


class DynamicObstacle(BaseMovingObstacle):
    def __init__(self, 
                 states:States3,
                 enveloppe:EnveloppeBase=None,
                 physics:PhysicsBase=None,
                 integrator:Integrator=None,
                 derivatives:TimeDerivatives3=None,
                 id:int=0
                 ):
        self._physics = physics or ZeroAccelerationPhysics()
        super().__init__(states=states, enveloppe=enveloppe, integrator=integrator, derivatives=derivatives, id=id)
    
    def update_derivatives(self, wind:WindVector, water:WaterVector, external_forces:GeneralizedForces):
        """
        Update the derivatives of the states.
        """
        self._derivatives, self._generalized_forces = self._physics.get_time_derivatives_and_forces(self._states, wind, water, external_forces)

    def rotation_matrix(self, angle:float, dim:int=2):
        """
        Get the rotation matrix.
        """
        return self._physics.rotation_matrix(angle, dim=dim) 

    @property
    def physics(self) -> PhysicsBase:
        return self._physics
    
class DynamicShip(DynamicObstacle):
    def __init__(self, 
                 states:States3,
                 path_to_physical_params:str=None,
                 length:float=None,
                 width:float=None,
                 ratio:float=None,
                 integrator:Integrator=None,
                 derivatives:TimeDerivatives3=None,
                 id:int=0
                 ):
        enveloppe = enveloppe or ShipEnveloppe(length=length, width=width, ratio=ratio)
        super().__init__(states=states, enveloppe=enveloppe, physics=ShipPhysics(path_to_physical_params), integrator=integrator, derivatives=derivatives, id=id)
    
class DriftingObstacle(DynamicObstacle):
    def __init__(self, 
                 states:States3,
                 enveloppe:EnveloppeBase=None,
                 integrator:Integrator=None,
                 derivatives:TimeDerivatives3=None,
                 id:int=0
                 ):
        super().__init__(states=states, enveloppe=enveloppe, physics=ZeroAccelerationPhysics(), integrator=integrator, derivatives=derivatives, id=id)

class DriftingShip(DriftingObstacle):
    def __init__(self, 
                 states:States3,
                 enveloppe:EnveloppeBase=None,
                 integrator:Integrator=None,
                 derivatives:TimeDerivatives3=None,
                 id:int=0
                 ):
        super().__init__(states=states, enveloppe=enveloppe, integrator=integrator, derivatives=derivatives, id=id)

# TODO: How do we make a sailing ship follow a path? This is the main objective of it...
class SailingShip(DynamicObstacle):
    def __init__(self, 
                 states:States3,
                 length:float=None,
                 width:float=None,
                 ratio:float=None,
                 integrator:Integrator=None,
                 derivatives:TimeDerivatives3=None,
                 id:int=0
                 ):
        enveloppe = enveloppe or ShipEnveloppe(length=length, width=width, ratio=ratio)
        super().__init__(states=states, enveloppe=enveloppe, physics=ZeroAccelerationSailingShipPhysics(), integrator=integrator, derivatives=derivatives, id=id)
    
class StaticObstacle(DriftingObstacle):
    def __init__(self, 
                 x:float,
                 y:float,
                 psi_def:float,
                 enveloppe:EnveloppeBase=None,
                 id:int=0
                 ):
        super().__init__(states=States3(x, y, psi_def), enveloppe=enveloppe, derivatives=TimeDerivatives3(), id=id)


def test():
    pass

if __name__ == "__main__":
    test()

# self._physics = physics or ZeroAccelerationPhysics()


# class ObstacleWithDynamics(BaseMovingObstacle):
#     def __init__(self, 
#                  states:States3,
#                  physics:phy.ShipPhysics=None,
#                  integrator:Integrator=None,
#                  derivatives:TimeDerivatives3=None,
#                  id:str="ObstacleWithDynamics"
#                  ):
#         super().__init__(states=states, physics=physics, integrator=integrator, derivatives=derivatives, id=id)
    
#     def update_derivatives(self, wind:WindVector, water:WaterVector, external_forces:GeneralizedForces):
#         """
#         Update the derivatives of the states.
#         """
#         self._derivatives = self._physics.derivatives(self._states, wind, water, external_forces)