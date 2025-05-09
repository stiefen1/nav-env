"""
We consider an interceptor (own ship), a (possibly moving) target, 
and a reference point (previous wpt for instance). 

v_d: Interceptor speed
v_t: Target speed
w_d: Interceptor point
w_r: Reference point
w_t: Target point

"""

from math import atan2, pi, sqrt
import numpy as np
from nav_env.control.path import Waypoints
from nav_env.control.guidance import Guidance
from abc import ABC, abstractmethod


class LOSLookAhead(Guidance):
    def __init__(self,
                 waypoints: Waypoints,
                 current_wpt_idx:int,
                 radius_of_acceptance:float=50,
                 kp:float=3.5e-2,
                 ki:float=0.,
                 *args,
                 **kwargs):
        super().__init__(waypoints=waypoints, current_wpt_idx=current_wpt_idx, radius_of_acceptance=radius_of_acceptance, *args, **kwargs)
        self._kp = kp
        self._ki = ki

    def get_desired_heading(self, x, y, *args, degree=False, **kwargs):
        if self.within_radius_of_acceptance(x, y):
            self.next_waypoint()
        next_wpt = self.get_next_waypoint()
        prev_wpt = self.get_prev_waypoint()
        convert_unit = 180/pi if degree else 1
        return convert_unit*LOS_lookahead(x, y, prev_wpt, next_wpt, *args, kp=self._kp, **kwargs)
    

class LOSLoopEnclosure(Guidance):
    def __init__(self, waypoints: Waypoints, radius:float, *args, current_wpt_idx:int=0, radius_of_acceptance:float=50, **kwargs):
        super().__init__(waypoints=waypoints, current_wpt_idx=current_wpt_idx, radius_of_acceptance=radius_of_acceptance, *args, **kwargs)
        self._radius = radius

    def get_desired_heading(self, x, y, *args, degree=False, **kwargs):
        if self.within_radius_of_acceptance(x, y):
            self.next_waypoint()
        next_wpt = self.get_next_waypoint()
        prev_wpt = self.get_prev_waypoint()
        convert_unit = 180/pi if degree else 1
        return convert_unit*LOS_enclosure(x, y, prev_wpt, next_wpt, *args, R=self._radius, **kwargs)

def LOS_enclosure(x, y, w_prev, w_next, *args, R:float=50, **kwargs):
    dwx = w_next[0] - w_prev[0]
    dwy = w_next[1] - w_prev[1]

    # For abs(dwx) > 0
    if dwy != 0:
        d = dwx / dwy
        f, e = w_prev
        g = f - d*e
        
        # Viète
        a = 1 + d**2
        b = 2*(d*g - d*x - y)
        c = x**2 + y**2 + g**2 - 2*g*x - R**2

        # Discriminant
        Delta = sqrt(b**2 - 4*a*c)

        # Get LOS point
        if dwy > 0:
            y_los = (-b + Delta) / (2*a)
        elif dwy < 0:
            y_los = (-b - Delta) / (2*a)
        
        x_los = w_prev[0] + d * (y_los - e)

    elif dwy == 0:
        assert dwx != 0, f"Current and previous waypoints must be different"

        # dwy == 0 means y_prev = y_next = y_los
        y_los = w_next[1]

        # Solve for x_los
        Delta = sqrt(R**2 - (y_los - y)**2)

        if dwx > 0:
            x_los = x + Delta
        elif dwx < 0:
            x_los = x - Delta

    # Desired angle
    psi_d = -atan2(x_los-x, y_los-y)

    return x_los, y_los, psi_d*180/pi

def LOS_lookahead(x, y, w_prev, w_next, *args, kp:float=3.5e-2, **kwargs):
    """
    psi_d(e) = psi_p + psi_r(e) = alpha_k + atan(-Kp*e)
    """
    
    # Compute psi_p
    psi_p = -atan2(w_next[0] - w_prev[0], w_next[1] - w_prev[1]) * 180 / pi

    # Compute e
    w_p = np.array(w_prev).T
    w_n = np.array(w_next).T
    w_unit = (w_p - w_n) / np.linalg.norm(w_p-w_n)

    p = np.array([x, y]).T - w_n
    p_unit = p / np.linalg.norm(p)

    sign = 1
    if p_unit[1] > w_unit[1]:
        sign = -1

    angle = sign * np.arccos(p_unit.T @ w_unit)
    e = (np.linalg.norm(w_n - p) * np.sin(angle)).astype(float)
    psi_r = atan2(kp*e, 1) * 180 / pi

    print(kp*e, psi_p, psi_r)


    return psi_p + psi_r


def interactive():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.widgets import Slider

    global x, y

    x = 0.
    y = 0.

    R = 30
    w_prev = (-50, -10)
    w_next = (50, 50)

    # Figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax_x = fig.add_axes([0.1, 0.01, 0.8, 0.03])
    ax_y = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    x_slider = Slider(ax_x, 'X', -50, 50)
    y_slider = Slider(ax_y, 'Y', -50, 50)

    def update_x(new_x):
        global x
        x = new_x
        update()

    def update_y(new_y):
        global y
        y = new_y
        update()

    def update():
        # LOS
        x_los, y_los, psi_d = LOS_enclosure(x, y, w_prev, w_next, R=R)
        psi_d_lookahead = LOS_lookahead(x, y, w_prev, w_next)

        # Patches
        c = Circle((x, y), R, fill=False, edgecolor='blue', linestyle='--')
        wp = Circle(w_prev, 3, facecolor='red')
        wn = Circle(w_next, 3, facecolor='red')
        los = Circle((x_los, y_los), 3, facecolor='green')

        ax.cla()

        ax.add_patch(c)
        ax.add_patch(wp)
        ax.add_patch(wn)
        ax.add_patch(los)
        ax.plot(*zip(w_prev, w_next), '--', color='red')
        ax.scatter(x, y, color='blue')

        # Plot
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        ax.set_aspect('equal')
        ax.set_title(f"x:{x_los:.0f} , y:{y_los:.0f} , $\\psi_d$:{psi_d:.0f} | Lookahead: {psi_d_lookahead:.0f}")
    
    x_slider.on_changed(update_x)
    y_slider.on_changed(update_y)
    update()
    plt.show()
    plt.close()

if __name__=="__main__":
    interactive()
