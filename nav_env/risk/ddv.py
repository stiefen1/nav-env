from nav_env.environment.environment import NavigationEnvironment
from nav_env.ships.ship import Ship
from nav_env.obstacles.obstacles import ObstacleWithKinematics
from nav_env.risk.risk import RiskMetric
from nav_env.risk.utils import get_relative_position_and_speed
from nav_env.obstacles.obstacles import Ellipse
from math import cos, sin, pi


def get_ddv_terms(own_ship: ObstacleWithKinematics, target_ship: ObstacleWithKinematics) -> float:
    assert isinstance(target_ship.domain, Ellipse), "Target ship domain must be an ellipse" # Degree of domain violation is only handled for elliptical domains
    px_rel, py_rel, vx_rel, vy_rel = get_relative_position_and_speed(own_ship, target_ship)
    
    # Orientation of target ship, w.r.t to the x-axis (ship's heading angle is measured counter-clockwise from the y-axis)
    sin2:float = sin(target_ship.states.psi_rad + pi/2)
    cos2:float = cos(target_ship.states.psi_rad + pi/2)
    
    da_ts = target_ship.domain.da
    db_ts = target_ship.domain.db
    a_ts = target_ship.domain.a
    b_ts = target_ship.domain.b

    # Rotated center of the ellipse
    h:float = da_ts * cos2 + db_ts * sin2
    k:float = da_ts * sin2 - db_ts * cos2

    a1:float = cos2**2 / a_ts**2 + sin2**2 / b_ts**2
    b1:float = 2 * cos2 * sin2 * (1/a_ts**2 - 1/b_ts**2)
    c1:float = sin2**2 / a_ts**2 + cos2**2 / b_ts**2

    a2:float = a1 * h**2 + c1 * k**2 + h * k * b1 - 1
    b21:float = h * (2 * px_rel * a1 + b1 * py_rel) + k * (2 * py_rel * c1 + px_rel * b1)
    b22:float = 2 * h * vx_rel * a1 + h * vy_rel * b1 + 2 * k * vy_rel * c1 + k * vx_rel * b1
    c21:float = a1 * px_rel**2 + b1 * px_rel * py_rel + c1 * py_rel**2
    c22:float = 2 * a1 * px_rel * vx_rel + b1 * px_rel * vy_rel + b1 * py_rel * vx_rel + 2 * c1 * py_rel * vy_rel
    c23:float = a1 * vx_rel**2 + b1 * vx_rel * vy_rel + c1 * vy_rel**2

    d:float = b22**2 - 4 * a2 * c23
    e:float = 2 * b21 * b22 - 4 * a2 * c22
    f:float = b21**2 - 4 * a2 * c21

    return h, k, a1, b1, c1, a2, b21, b22, d, e, f

def get_t1_t2(b22:float, d:float, e:float, f:float) -> tuple[float, float]:
    denominator:float = 2 * (d**2 - b22**2 * d)

    under_root:float = (d * e - b22**2 * e)**2 - (d**2 - b22**2 * d) * (e**2 - 4 * b22**2 * f)
    if under_root < 0.0:
        under_root = 0.0

    numerator:float = b22**2 * e - d * e - under_root**0.5
    t1:float = numerator / denominator

    numerator:float = b22**2 * e - d * e + under_root**0.5
    t2:float = numerator / denominator
    return t1, t2

def get_approach_factors(t1, t2, a2, b21, b22, d, e, f) -> tuple[float, float, float, float]:
    under_root = d * pow(t1,2) + e * t1 + f
    if (under_root < 0.0):
        under_root = 0.0
    f1:float = (-b21 - b22 * t1 - under_root**0.5) / (2 * a2)
    if (f1 < -0.01):
        f1 = 1000000
    f2:float = (-b21 - b22 * t1 + under_root**0.5) / (2 * a2)
    if (f2 < -0.01):
        f2 = 1000000

    under_root = d * pow(t2,2) + e * t2 + f
    if (under_root < 0.0):
        under_root = 0.0
    f3:float = (-b21 - b22 * t2 - under_root**0.5) / (2 * a2)
    if (f3 < -0.01):
        f3 = 1000000
    f4:float = (-b21 - b22 * t2 + under_root**0.5) / (2 * a2)
    if (f4 < -0.01):
        f4 = 1000000

    return f1, f2, f3, f4

def get_min_approach_factor(own_ship: ObstacleWithKinematics, target_ship: ObstacleWithKinematics) -> float:
    """
    Calculate the minimum approach factor of the target ship by the own ship.
    """
    if not isinstance(target_ship.domain, Ellipse): # Degree of domain violation is only handled for elliptical domains
        return float("NaN")

    h, k, a1, b1, c1, a2, b21, b22, d, e, f = get_ddv_terms(own_ship, target_ship)
    t1, t2 = get_t1_t2(b22, d, e, f)
    f1, f2, f3, f4 = get_approach_factors(t1, t2, a2, b21, b22, d, e, f)

    fmin:float = min([f1, f2, f3, f4]) # Minimum approach factor
    return fmin

def get_time_min_approach_factor(own_ship: ObstacleWithKinematics, target_ship: ObstacleWithKinematics) -> float:
    if not isinstance(target_ship.domain, Ellipse): # Degree of domain violation is only handled for elliptical domains
        return float("NaN")

    h, k, a1, b1, c1, a2, b21, b22, d, e, f = get_ddv_terms(own_ship, target_ship)
    t1, t2 = get_t1_t2(b22, d, e, f)
    f1, f2, f3, f4 = get_approach_factors(t1, t2, a2, b21, b22, d, e, f)
    
    fmin:float = min([f1, f2, f3, f4]) # Minimum approach factor
    if (abs(fmin - f1) < 1e-6) or (abs(fmin - f2) < 1e-6):
        tmin = t1
    else:
        tmin = t2
    return tmin

def get_ddv(own_ship: ObstacleWithKinematics, target_ship: ObstacleWithKinematics) -> float:
    """
    Calculate the Degree of Domain Violation of the target ship by the own ship.
    """
    if not isinstance(target_ship.domain, Ellipse): # Degree of domain violation is only handled for elliptical domains
        return float("NaN")

    fmin:float = get_min_approach_factor(own_ship, target_ship)
    ddv:float = max(0.0, 1.0 - fmin)
    return ddv

def get_tdv(own_ship: ObstacleWithKinematics, target_ship: ObstacleWithKinematics) -> float:
    """
    Calculate the Time to Domain Violation of the target ship by the own ship.
    """
    if not isinstance(target_ship.domain, Ellipse): # Degree of domain violation is only handled for elliptical domains
        return float("NaN")
    
    px_rel, py_rel, vx_rel, vy_rel = get_relative_position_and_speed(own_ship, target_ship)

    h, k, a1, b1, c1, a2, b21, b22, d, e, f = get_ddv_terms(own_ship, target_ship)
    t1, t2 = get_t1_t2(b22, d, e, f)
    f1, f2, f3, f4 = get_approach_factors(t1, t2, a2, b21, b22, d, e, f)
    fmin:float = min([f1, f2, f3, f4]) # Minimum approach factor
    
    xe:float = px_rel + h
    ye:float = py_rel + k
    a3:float = a1 * vx_rel**2 + b1 * vx_rel * vy_rel + c1 * vy_rel**2
    b3:float = 2 * (a1 * xe * vx_rel + c1 * ye * vy_rel) + b1 * (xe * vy_rel + ye * vx_rel)
    c3:float = a1 * xe**2 + b1 * xe * ye + c1 * ye**2 - 1
    under_root = b3**2 - 4 * a3 * c3
    if under_root < 0.0:
        under_root = 0.0
    if fmin >= 1:
        under_root = 0.0
    tdv1:float = (-b3 - under_root**0.5) / (2 * a3)
    tdv2:float = (-b3 + under_root**0.5) / (2 * a3)
    tdv:float = min(tdv1, tdv2)
    return tdv


class DDV(RiskMetric):
    def __init__(self, env:NavigationEnvironment):
        super().__init__(env)

    def calculate(self, ship:Ship, **kwargs) -> float:
        max_ddv = 0.

        for moving_obstacle in self.env.obstacles:
            if isinstance(moving_obstacle.domain, Ellipse):
                ddv = get_ddv(ship, moving_obstacle)
                if ddv > max_ddv:
                    max_ddv = ddv


        for target_ship in self.env.target_ships:
            if isinstance(target_ship.domain, Ellipse): 
                ddv = get_ddv(ship, target_ship)
                if ddv > max_ddv:
                    max_ddv = ddv

                
        # print(f"{100*max_ddv:.2f}")
        return 100 * max_ddv # 100 * max_ddv # min_dcpa # min_tcpa # min_tdv # 100*max_ddv
    
    def plot(self, ax=None, **kwargs):
        pass

class TDV(RiskMetric):
    def __init__(self, env:NavigationEnvironment):
        super().__init__(env)

    def calculate(self, ship:Ship, **kwargs) -> float:
        min_tdv = float("inf")

        for moving_obstacle in self.env.obstacles:
            if isinstance(moving_obstacle.domain, Ellipse):
                tdv = get_tdv(ship, moving_obstacle)
                if tdv < min_tdv:
                    min_tdv = tdv

        for target_ship in self.env.target_ships:
            if isinstance(target_ship.domain, Ellipse):
                tdv = get_tdv(ship, target_ship)
                if tdv < min_tdv:
                    min_tdv = tdv

        # print(f"{min_tdv:.2f}")
        return min_tdv
    
    def plot(self, ax=None, **kwargs):
        pass
    

    
def test():
    from nav_env.ships.ship import SimpleShip
    from nav_env.ships.collection import ShipCollection
    # Test the DDV class
    ship1 = SimpleShip(None, None, "1")
    ship2 = SimpleShip(None, None, "2")
    ship3 = SimpleShip(None, None, "3")
    target_ships = ShipCollection([ship1, ship2, ship3])

    ship4 = SimpleShip(None, None, "4")
    ship5 = SimpleShip(None, None, "5")
    ship6 = SimpleShip(None, None, "6")
    own_ships = ShipCollection([ship4, ship5, ship6])

    env = NavigationEnvironment(own_ships=own_ships, target_ships=target_ships)
    ddv = DDV(env)
    ddv.calculate(0)


if __name__ == "__main__":
    test()
    


    

