from abc import ABC, abstractmethod
from nav_env.ships.states import ShipStates3, ShipTimeDerivatives3
from nav_env.geometry.vector import Vector
from nav_env.ships.params import ShipPhysicalParams, ShipPhysics
from nav_env.obstacles.ship import ShipEnveloppe
from nav_env.obstacles.collection import ObstacleCollection
from nav_env.simulation.integration import Integrator, Euler
from nav_env.control.command import GeneralizedForces
from nav_env.control.controller import ControllerBase, Controller
from nav_env.environment.disturbances import DisturbanceCollection

class ShipWithDynamicsBase(ABC):
    def __init__(self,
                 states:ShipStates3,
                 physics:ShipPhysics=ShipPhysics(),
                 controller:ControllerBase=Controller(),
                 integrator:Integrator=Euler(),
                 derivatives:ShipTimeDerivatives3=ShipTimeDerivatives3(),
                 name:str="ShipWithDynamicsBase"
                 ):
        self._states = states
        self._physics = physics
        self._controller = controller
        self._enveloppe = ShipEnveloppe(length=physics.length, width=physics.width)
        self._integrator = integrator
        self._derivatives = derivatives
        self._name = name
        self._dx = None # Initialize differential to None

    def draw(self):
        """
        Draw the ship for pygame.
        """
        pass

    def plot(self, **kwargs):
        """
        Plot the ship.
        """
        return self._enveloppe.plot(**kwargs)

    def step(self, disturbances:DisturbanceCollection=DisturbanceCollection(), external_forces:GeneralizedForces=GeneralizedForces()):
        """
        Step the ship.
        """
        self.update_derivatives(disturbances, external_forces)
        self.integrate()
        self.update_enveloppe()
        
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

    def collide(self, obstacle:ObstacleCollection) -> bool:
        """
        Check if the ship collides with an obstacle.
        """
        for obs in obstacle:
            if self._enveloppe.intersects(obs):
                return True
        return False

    @abstractmethod
    def update_derivatives(self, disturbances:DisturbanceCollection, external_forces:GeneralizedForces):
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
    def derivatives(self) -> ShipTimeDerivatives3:
        return self._derivatives
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def position(self) -> tuple:
        return self._states.x, self._states.y
    

class SimpleShip(ShipWithDynamicsBase):
    def __init__(self,
                 states:ShipStates3 = ShipStates3(),
                 physics:ShipPhysics = ShipPhysics(),
                 controller:ControllerBase=Controller(),
                 integrator:Integrator=Euler(),
                 derivatives:ShipTimeDerivatives3=ShipTimeDerivatives3(), 
                 name:str="SimpleShip"
                 ):
        super().__init__(states=states, physics=physics, controller=controller, integrator=integrator, derivatives=derivatives, name=name)

    def update_derivatives(self, disturbances:DisturbanceCollection, external_forces:GeneralizedForces):
        """
        Update the derivatives of the states.
        """
        # x_dot = A * x, no acceleration
        self._derivatives.x_dot = self._states.x_dot
        self._derivatives.y_dot = self._states.y_dot
        self._derivatives.psi_dot = self._states.psi_dot
        self._derivatives.x_dot_dot = 0.
        self._derivatives.y_dot_dot = 0.
        self._derivatives.psi_dot_dot = 0.

class Ship(ShipWithDynamicsBase):
    def __init__(self, 
                 states:ShipStates3 = ShipStates3(),
                 physics:ShipPhysics = ShipPhysics(),
                 controller:ControllerBase=Controller(),
                 integrator:Integrator=Euler(),
                 derivatives:ShipTimeDerivatives3=ShipTimeDerivatives3(), 
                 name:str="Ship"):
        super().__init__(states=states, physics=physics, controller=controller, integrator=integrator, derivatives=derivatives, name=name)

    def update_derivatives(self, disturbances:DisturbanceCollection, external_forces:GeneralizedForces):
        """
        Update the derivatives of the states.
        """
        self._derivatives = self._physics.get_time_derivatives(self._states)

def test():
    import matplotlib.pyplot as plt
    from nav_env.viz.matplotlib_screen import MatplotlibScreen as Screen
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.ships.collection import ShipCollection

    dt = 0.03
    x0 = ShipStates3(0., 0., 0., 1., 2., 10)
    ship = Ship(x0, integrator=Euler(dt))
    xlim, ylim = (-300, -300), (300, 300)
    env = Env(own_ships=ShipCollection([ship]))
    screen = Screen(env, lim=(xlim, ylim))
    screen.play(dt=dt, tf=20)

    # ax = ship.plot()
    # ax.set_xlim(*xlim)
    # ax.set_ylim(*ylim)
    # plt.waitforbuttonpress()
    # t = 0
    # while t<10:
    #     action = GeneralizedForces() # TODO: Add wind / water influence 
    #     ship.step(action)
    #     t += dt
    #     ax.cla()
    #     ship.plot(ax=ax)
    #     ax.set_xlim(*xlim)
    #     ax.set_ylim(*ylim)
    #     plt.pause(dt)
    #     if t % 1 < dt:
    #         print(f"{t:.2f} - {ship}")
            

        

if __name__ == "__main__":
    test()
    
    