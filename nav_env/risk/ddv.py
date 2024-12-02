from nav_env.environment.environment import NavigationEnvironment
from nav_env.risk.risk import RiskMetric
from copy import deepcopy

#### Joanna's code ####

import math

# function get_fmin_DDV_TDV to compute fmin, DDV and TDV according to "An analysis of domain-based ship collision risk parameters" http://dx.doi.org/10.1016/j.oceaneng.2016.08.030
# INPUT PARAMETERS:
# x1 - own ship (OS) real position X [NM or m]
# y1 - own ship (OS) real position Y [NM or m]
# x2 - target ship (TS) real position X [NM or m]
# y2 - target ship (TS) real position Y [NM or m]
# v1x - X vector of own ship (OS) speed [kn or m/s]
# v1y - Y vector of own ship (OS) speed [kn or m/s]
# v2x - X vector of target ship (TS) speed [kn or m/s]
# v2y - Y vector of target ship (TS) speed [kn or m/s]
# decentralised elliptic ship domain parameters:
# a - semi-major axis [NM or m]
# b - semi-minor axis [NM or m]
# da - a ship's displacement from the ellipse's centre towards aft along the semi-major axis [NM or m]
# db - a ship's displacement from the ellipse's centre towards port along the semi-minor axis [NM or m]
#
# OUTPUT RESULTS: a tuple of
# fmin [-]
# ddv [-]
# tdv [h or sec]
# tde [h or sec]
# dcpa [NM or m]
# tcpa [h or sec]
# tmin [h or sec]
# h or sec - depends if we use NM (then h) or m/s (then sec)

def get_fmin_DDV_TDV(x1:float,y1:float,x2:float,y2:float,v1x:float,v1y:float,v2x:float,v2y:float, a:float, b:float, da:float, db:float) -> tuple[float,float,float,float,float,float,float]:
    v1:float = math.sqrt(pow(v1x,2)+ pow(v1y,2))
    v2:float = math.sqrt(pow(v2x,2)+ pow(v2y,2))
    if (v2==0.0):
        return float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"),  float("NaN")   # if ship2 does not move then fmin, DDV, TDV, TDE, DCPA, TCPA are undefined

    sin2:float = float(v2y / v2)
    cos2:float = float(v2x / v2)
    x:float = float(x2 - x1)
    y:float = float(y2 - y1)
    vx:float = float(v2x - v1x)
    vy:float = float(v2y - v1y)

    # DCPA & TCPA computations
    vr:float = math.sqrt(pow(vx,2)+pow(vy,2))
    dcpa:float = abs ((x*vy - y*vx)/(vr))
    tcpa:float = -(x*vx + y*vy)/(vr*vr)
    # end of DCPA & TCPA computations

    if (vx==0.0 and vy==0.0):
        return float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")  # if OS and TS have exact the same speed vectors then fmin, DDV, TDV, TDE, DCPA, TCPA are undefined

    h:float = da * cos2 + db * sin2
    k:float = da * sin2 - db * cos2

    a1:float = pow(cos2,2) / pow(a,2) + pow(sin2,2) / pow(b,2)
    b1:float = 2 * cos2 * sin2 * (1 / pow(a,2) - 1 / pow(b,2))
    c1:float = pow(cos2,2) / pow(b,2) + pow(sin2,2) / pow(a,2)

    a2:float = a1 * pow(h,2) + c1 * pow(k,2) + h * k * b1 - 1
    b21:float = h * (2 * x * a1 + b1 * y) + k * (2 * y * c1 + x * b1)
    b22:float = 2 * h * vx * a1 + h * vy * b1 + 2 * k * vy * c1 + k * vx * b1
    c21:float = a1 * pow(x,2) + b1 * x * y + c1 * pow(y,2)
    c22:float = 2 * a1 * x * vx + b1 * x * vy + b1 * y * vx + 2 * c1 * y * vy
    c23:float = a1 * pow(vx,2) + b1 * vx * vy + c1 * pow(vy,2)

    d:float = pow(b22,2) - 4 * a2 * c23
    e:float = 2 * b21 * b22 - 4 * a2 * c22
    f:float = pow(b21,2) - 4 * a2 * c21

    under_root:float = pow((d * e - pow(b22,2) * e),2) - (pow(d,2) - pow(b22,2) * d) * (pow(e,2) - 4 * pow(b22,2) * f)
    if (under_root < 0.0):
        under_root = 0.0
    nominator:float = pow(b22,2) * e - d * e - math.sqrt(under_root)
    denominator:float = float (2 * (pow(d,2) - pow(b22,2) * d))
    t1:float = nominator / denominator
    nominator = pow(b22,2) * e - d * e + math.sqrt(under_root)
    t2:float = nominator / denominator

    under_root = d * pow(t1,2) + e * t1 + f
    if (under_root < 0.0):
        under_root = 0.0
    f1:float = (-b21 - b22 * t1 - math.sqrt(under_root)) / (2 * a2)
    if (f1 < -0.01):
        f1 = 1000000
    f2:float = (-b21 - b22 * t1 + math.sqrt(under_root)) / (2 * a2)
    if (f2 < -0.01):
        f2 = 1000000

    under_root = d * pow(t2,2) + e * t2 + f
    if (under_root < 0.0):
        under_root = 0.0
    f3:float = (-b21 - b22 * t2 - math.sqrt(under_root)) / (2 * a2)
    if (f3 < -0.01):
        f3 = 1000000
    f4:float = (-b21 - b22 * t2 + math.sqrt(under_root)) / (2 * a2)
    if (f4 < -0.01):
        f4 = 1000000

    fmin:float = min(min(f1, f2), min(f3, f4))
    if (abs(fmin - f1) < 0.000001) or (abs(fmin - f2) < 0.000001):
        tmin = t1
    else:
        tmin = t2

    xe:float = x + da * cos2 + db * sin2
    ye:float = y + da * sin2 - db * cos2
    a3:float = a1 * pow(vx,2) + b1 * vx * vy + c1 * pow(vy,2)
    b3:float = 2 * (a1 * xe * vx + c1 * ye * vy) + b1 * (xe * vy + ye * vx)
    c3:float = a1 * pow(xe,2) + b1 * xe * ye + c1 * pow(ye,2) - 1
    under_root = pow(b3,2) - 4 * a3 * c3
    if (under_root < 0.0):
        under_root = 0.0
    if (fmin >= 1):
        under_root = 0.0
    tdv1:float = (-b3 - math.sqrt(under_root)) / (2 * a3)
    tdv2:float = (-b3 + math.sqrt(under_root)) / (2 * a3)
    tdv:float = min(tdv1, tdv2)
    tde:float = max(tdv1, tdv2)

    ddv:float = max(0.0, 1.0 - fmin)
    if (ddv == 0.0):
        tdv = float("NaN") # we set NaN (Not a Number) for TDV since there is no domain violation (DDV = 0.0)

    return (fmin, ddv, tdv, tde, dcpa, tcpa, tmin)


#### End of Joanna's code ####


class DDV(RiskMetric):
    def __init__(self, env:NavigationEnvironment):
        super().__init__(env)

    def calculate(self, t:float, **kwargs) -> float:
        target_ships_collection = self._env._target_ships
        own_ships_collection = self._env._own_ships

        for i, ship_i in enumerate(own_ships_collection):
            target_ships_i = deepcopy(target_ships_collection)
            target_ships_i.append(own_ships_collection.get_except(ship_i)) # Add all other ships except ship i
            for j, target_ship_ij in enumerate(target_ships_i): 
                print(f"ship {ship_i} and target {target_ship_ij}")
                # Compute ddv for ship ij
                pass

        ddv = 0.

        return ddv
    
    def plot(self, ax=None, **kwargs):
        pass

    def __repr__(self):
        return "DDV"
    
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
    


    

