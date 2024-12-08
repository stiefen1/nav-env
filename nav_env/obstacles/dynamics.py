from nav_env.obstacles.obstacles import Obstacle
from nav_env.ships.states import States3, TimeDerivatives3
from nav_env.control.states import DeltaStates
from nav_env.control.command import GeneralizedForces
from nav_env.simulation.integration import Integrator, Euler
from nav_env.obstacles.ship import Enveloppe 
import nav_env.ships.physics as phy
import pygame

from copy import deepcopy


class BaseMovingObstacle(Obstacle):
    def __init__(self,
                 states:States3,
                 physics:phy.ShipPhysics=None,
                 integrator:Integrator=None,
                 derivatives:TimeDerivatives3=None,
                 name:str="ShipWithDynamicsBase"
                 ):
        self._states = states
        self._initial_states = deepcopy(states)
        self._physics = physics or phy.ShipPhysics()
        self._integrator = integrator or Euler()
        self._derivatives = derivatives or TimeDerivatives3()
        self._enveloppe = enveloppe or Enveloppe()
        self._name = name
        self._dx = None # Initialize differential to None
        self._accumulated_dx = DeltaStates(0., 0., 0., 0., 0., 0.) # Initialize accumulated differential to 0
        self._generalized_forces = GeneralizedForces() # Initialize generalized forces acting on the ship to 0


    def reset(self):
        """
        Reset the ship to its initial state.
        """
        self._states = deepcopy(self._initial_states)
        self._enveloppe = self.get_initial_enveloppe()
        self._dx = None
        self._generalized_forces = GeneralizedForces()

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

        # print(self.name, self._physics.generalized_forces.f_x / self._derivatives.x_dot_dot, self._physics.generalized_forces.f_y / self._derivatives.y_dot_dot)
        return ax
    
    def plot_frame(self, ax=None, **kwargs):
        """
        Plot the ship frame.
        """
        if ax is None:
            _, ax = plt.subplots()

        R = self._physics.rotation_matrix(self._states.psi_rad, dim=2)
        ax.quiver(*self._states.xy, *R[0, :], color='r', **kwargs)
        ax.quiver(*self._states.xy, *R[1, :], color='g', **kwargs)
        return ax
    
    def draw_frame(self, screen, *args, scale=1, **kwargs):
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

    def update_enveloppe(self):
        if self._dx is not None:
            self._enveloppe = self._enveloppe.rotate_and_translate(self._dx[0], self._dx[1], self._dx[2])
        else:
            raise UserWarning(f"self._dx is None, you must first call integrate()")

    def update_enveloppe_from_accumulation(self):
        """
        Update the enveloppe from the accumulated differential.

        WARNING!!! DONT USE THIS METHOD UNLESS YOU KNOW EXACTLY WHAT YOU ARE DOING
        """
        self._enveloppe = self._enveloppe.rotate_and_translate(self._accumulated_dx[0], self._accumulated_dx[1], self._accumulated_dx[2])
        self._accumulated_dx = DeltaStates(0., 0., 0., 0., 0., 0.)

    def collide(self, obstacle:ObstacleCollection) -> bool:
        """
        Check if the ship collides with an obstacle.
        """
        for obs in obstacle:
            if self._enveloppe.intersects(obs):
                return True
        return False

    @abstractmethod
    def update_derivatives(self, wind:WindVector, water:WaterVector, external_forces:GeneralizedForces):
        """
        Update the derivatives of the states.
        """
        # MAYBE WE SHOULD ONLY CONSIDER GENERALIZED FORCE TO SIMPLIFY AT FIRST
        pass

    def __repr__(self):
        return f"{type(self).__name__}({self._states.__dict__})"
    
    @property
    def states(self) -> ShipStates3:
        return self._states
    
    @property
    def enveloppe(self) -> ShipEnveloppe:
        return self._enveloppe
    
    @property
    def physical_params(self) -> ShipPhysicalParams:
        return self._physics.params
    
    @property
    def integrator(self) -> Integrator:
        return self._integrator
    
    @property
    def derivatives(self) -> TimeDerivatives3:
        return self._derivatives
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def position(self) -> tuple:
        return self._states.x, self._states.y
    
    @property
    def physics(self) -> phy.ShipPhysics:
        return self._physics
    
    @property
    def generalized_forces(self) -> GeneralizedForces:
        """
        Get the generalized forces acting on the ship.
        """
        return self._generalized_forces
    

class ObstacleWithDynamics(BaseMovingObstacle):
    def __init__(self, 
                 states:States3,
                 physics:phy.ShipPhysics=None,
                 integrator:Integrator=None,
                 derivatives:TimeDerivatives3=None,
                 name:str="ObstacleWithDynamics"
                 ):
        super().__init__(states=states, physics=physics, integrator=integrator, derivatives=derivatives, name=name)
    
    def update_derivatives(self, wind:WindVector, water:WaterVector, external_forces:GeneralizedForces):
        """
        Update the derivatives of the states.
        """
        self._derivatives = self._physics.derivatives(self._states, wind, water, external_forces)