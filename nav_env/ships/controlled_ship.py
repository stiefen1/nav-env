from nav_env.ships.ship import ShipWithDynamicsBase
from nav_env.ships.states import States3
from nav_env.control.states import DeltaStates
import nav_env.ships.physics as phy
from nav_env.control.command import GeneralizedForces
from nav_env.simulation.integration import Integrator, Euler
from nav_env.ships.states import States3, TimeDerivatives3
from nav_env.control.controller import ControllerBase, Controller
from nav_env.obstacles.obstacles import Obstacle

from copy import deepcopy

class ShipWithControl(ShipWithDynamicsBase):
    def __init__(
            self,
            states:States3,
            controller:ControllerBase=None,
            physics:phy.ShipPhysics=None,
            integrator:Integrator=None,
            derivatives:TimeDerivatives3=None,
            name:str="ShipWithDynamicsBase",
            domain:Obstacle=None,
            domain_margin_wrt_enveloppe:float=0.,
            id:int=None,
            **kwargs
                 ):
        self._states = states
        self._initial_states = deepcopy(states)
        self._physics = physics or phy.ShipPhysics()
        self._controller = controller or Controller()
        self._integrator = integrator or Euler()
        self._derivatives = derivatives or TimeDerivatives3()
        self._dx = None # Initialize differential to None
        self._accumulated_dx = DeltaStates(0., 0., 0., 0., 0., 0.) # Initialize accumulated differential to 0
        self._generalized_forces = GeneralizedForces() # Initialize generalized forces acting on the ship to 0

        super().__init__(
            states=states,
            length=self._physics.length,
            width=self._physics.width,
            dt=self._integrator.dt,
            domain=domain,
            domain_margin_wrt_enveloppe=domain_margin_wrt_enveloppe,
            name=name,
            id=id
        )