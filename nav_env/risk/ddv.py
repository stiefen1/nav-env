from nav_env.environment.environment import NavigationEnvironment
from nav_env.ships.ship import Ship
from nav_env.obstacles.obstacles import MovingObstacle
from nav_env.risk.risk import RiskMetric
from nav_env.risk.utils import get_relative_position_and_speed
from nav_env.obstacles.obstacles import Ellipse
from math import cos, sin, pi


def get_ddv_terms(own_ship: MovingObstacle, target_ship: MovingObstacle) -> float:
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

def get_min_approach_factor(own_ship: MovingObstacle, target_ship: MovingObstacle) -> float:
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

def get_time_min_approach_factor(own_ship: MovingObstacle, target_ship: MovingObstacle) -> float:
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

def get_ddv(own_ship: MovingObstacle, target_ship: MovingObstacle) -> float:
    """
    Calculate the Degree of Domain Violation of the target ship by the own ship.
    """
    if not isinstance(target_ship.domain, Ellipse): # Degree of domain violation is only handled for elliptical domains
        return float("NaN")

    fmin:float = get_min_approach_factor(own_ship, target_ship)
    ddv:float = max(0.0, 1.0 - fmin)
    return ddv

def get_tdv(own_ship: MovingObstacle, target_ship: MovingObstacle) -> float:
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

def get_current_ddv(obstacle: MovingObstacle, x:float, y:float) -> float:
    """
    Calculate the current Degree of Domain Violation of an object located at x, y
    """
    if not isinstance(obstacle.domain, Ellipse): # Degree of domain violation is only handled for elliptical domains
        return float("NaN")

    da_ts = obstacle.domain.da
    db_ts = obstacle.domain.db
    a_ts = obstacle.domain.a
    b_ts = obstacle.domain.b

    # print(f"da: {da_ts}, db: {db_ts}, a: {a_ts}, b: {b_ts}")

    # Place center in the ellipse frame
    x -= (obstacle.states.x)
    y -= (obstacle.states.y)

    # Rotate the point to the ellipse frame
    sin2:float = sin(obstacle.states.psi_rad + pi/2)
    cos2:float = cos(obstacle.states.psi_rad + pi/2)

    x_rot:float = x * cos2 + y * sin2
    y_rot:float = x * sin2 - y * cos2

    # Calculate the degree of domain violation
    fmin:float = ((x_rot - da_ts) / a_ts)**2 + ((y_rot - db_ts) / b_ts)**2 # If this equals 1, no violation

    # print(f"Rotated point: ({x_rot}, {y_rot}), fmin: {fmin}, x: {x}, y: {y}, a: {a_ts}, b: {b_ts}, da: {da_ts}, db: {db_ts}")

    ddv = max(0.0, 1.0 - fmin)
    return ddv


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
    

class DDV2(RiskMetric):
    def __init__(self, env:NavigationEnvironment):
        super().__init__(env)

    def calculate(self, ship:Ship, x:float=None, y:float=None, **kwargs) -> float:
        if not isinstance(ship.domain, Ellipse):
            return None
        
        max_ddv = 0.

        if x is not None and y is not None:
            max_ddv = get_current_ddv(ship, x, y)
        else:
            for moving_obstacle in self.env.obstacles:
                ddv = get_current_ddv(ship, moving_obstacle.states.x, moving_obstacle.states.y)
                if ddv > max_ddv:
                    max_ddv = ddv

            for target_ship in self.env.target_ships:
                ddv = get_current_ddv(ship, target_ship.states.x, target_ship.states.y)
                if ddv > max_ddv:
                    max_ddv = ddv

            for own_ship in self.env.own_ships.get_except(ship):
                ddv = get_current_ddv(ship, own_ship.states.x, own_ship.states.y)
                if ddv > max_ddv:
                    max_ddv = ddv

        return 100 * max_ddv
    
    def plot(self, ship:Ship, ax=None, **kwargs):
        super().plot(ship, ax=ax, **kwargs)

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
    from nav_env.ships.states import States3
    import matplotlib.pyplot as plt

    lim = 200

    # Test the DDV class
    ship1 = SimpleShip(States3(-50, 30, -150), None, name="1", domain=Ellipse(50, -30, 100, 50))
    own_ships = [ship1]
    ship2 = SimpleShip(States3(100, 100), None, name="2")
    target_ships = [ship2]

    env = NavigationEnvironment(
        own_ships=own_ships,
        target_ships=target_ships
        )
    
    ddv = DDV2(env)
    ddv.plot(env.own_ships[0], colorbar=True)
    plt.show()


if __name__ == "__main__":
    test()
    


    

