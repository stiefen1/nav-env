from nav_env.risk.risk import RiskMetric
from nav_env.ships.ship import SimpleShip
from nav_env.environment.environment import NavigationEnvironment

class TTG(RiskMetric):
    def __init__(self, env:NavigationEnvironment):
        super().__init__(env)
    
    def calculate(self, ship:SimpleShip, t_max:float=100., dt:float=1., **kwargs) -> float:
        """
        Calculate the Time To Grounding (TTG) for a ship.
        """
        t:float = 0.
        while t < t_max:
            if ship.collide(self.env.shore):
                return t
            
            ship.step(dt)
            t += dt
        return t_max
    
    def plot(self, ax=None, **kwargs):
        pass
    
    def __repr__(self):
        return "TTG"