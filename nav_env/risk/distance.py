from nav_env.risk.risk import RiskMetric
from nav_env.ships.ship import Ship

class Distance(RiskMetric):
    def __init__(self, env):
        super().__init__(env)
        
    def calculate(self, ship:Ship) -> float:
        """
        Calculate the distance between two ships.
        """
        dist = []
        for target in self._env.target_ships.get_except(ship):
            dist.append(ship._enveloppe.distance(target._enveloppe))
        for ships in self._env.own_ships.get_except(ship):
            dist.append(ship._enveloppe.distance(ships._enveloppe))
        for obs in self._env.obstacles:
            dist.append(ship._enveloppe.distance(obs._geometry))
        return min(dist)
    
    def plot(self, ax=None, **kwargs):
        pass