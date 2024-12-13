from nav_env.obstacles.obstacles import ObstacleWithKinematics
from nav_env.risk.utils import get_relative_position_and_speed
from nav_env.risk.risk import RiskMetric
from nav_env.environment.environment import NavigationEnvironment


def get_tcpa(own_ship: ObstacleWithKinematics, target_ship: ObstacleWithKinematics) -> float:
    """
    Calculate the Degree of Domain Violation of the target ship by the own ship.
    """
    px_rel, py_rel, vx_rel, vy_rel = get_relative_position_and_speed(own_ship, target_ship)
    
    # Norm of relative speed vector
    v_rel_squared = (vx_rel**2 + vy_rel**2)
    if v_rel_squared == 0:
        return -float("inf")

    tcpa:float = -(px_rel * vx_rel + py_rel * vy_rel)/(v_rel_squared)
    return tcpa

def get_dcpa(own_ship: ObstacleWithKinematics, target_ship: ObstacleWithKinematics) -> float:
    """
    Calculate the Degree of Domain Violation of the target ship by the own ship.
    """
    px_rel, py_rel, vx_rel, vy_rel = get_relative_position_and_speed(own_ship, target_ship)
    
    # Norm of relative speed vector
    v_rel_squared = (vx_rel**2 + vy_rel**2)
    if v_rel_squared == 0: # if OS and TS have exact the same speed vectors then DCPA is undefined (0/0)
        return float("NaN")

    dcpa:float = abs( (px_rel * vy_rel - py_rel * vx_rel) / (v_rel_squared**0.5) )
    return dcpa

class TCPA(RiskMetric):
    def __init__(self, env:NavigationEnvironment):
        super().__init__(env)
        
    def calculate(self, ship:ObstacleWithKinematics) -> float:
        """
        Calculate the Time to Closest Point of Approach of the target ship by the own ship.
        """
        min_tcpa = float("inf")
        for target_ship in self._env.target_ships:
            tcpa = get_tcpa(ship, target_ship)
            if tcpa < min_tcpa:
                min_tcpa = tcpa

        for moving_obstacle in self._env.obstacles:
            tcpa = get_tcpa(ship, moving_obstacle)
            if tcpa < min_tcpa:
                min_tcpa = tcpa
        return min_tcpa
    
    def plot(self, ax=None, **kwargs):
        pass

class DCPA(RiskMetric):
    def __init__(self, env:NavigationEnvironment):
        super().__init__(env)
        
    def calculate(self, ship:ObstacleWithKinematics) -> float:
        """
        Calculate the Distance to Closest Point of Approach of the target ship by the own ship.
        """
        min_dcpa = float("inf")
        for target_ship in self._env.target_ships:
            dcpa = get_dcpa(ship, target_ship)
            if dcpa < min_dcpa:
                min_dcpa = dcpa

        for moving_obstacle in self._env.obstacles:
            dcpa = get_dcpa(ship, moving_obstacle)
            if dcpa < min_dcpa:
                min_dcpa = dcpa
        return min_dcpa
    
    def plot(self, ax=None, **kwargs):
        pass