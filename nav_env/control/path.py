from shapely import LineString
from nav_env.geometry.wrapper import GeometryWrapper
from nav_env.obstacles.obstacles import Obstacle
import matplotlib.pyplot as plt, matplotlib.colors as mat_colors
import warnings, sqlite3
from datetime import datetime, timedelta, time
from math import atan2, pi
from typing import Any
TIME_FORMAT = "%d-%m-%Y %H:%M:%S"

class Waypoints(GeometryWrapper):
    def __init__(self, waypoints: list = None):
        self._waypoints = [] if waypoints is None else waypoints
        super().__init__(xy=self._waypoints, geometry_type=LineString)

    def __iter__(self):
        return iter(self.get_xy_as_list())
    
    def __getitem__(self, idx):
        xy = self.get_xy_as_list()
        assert idx < len(xy), "Index out of range."
        return xy[idx]
    
    def __call__(self, value:float) -> tuple[float, float]:
        if value > 1.:
            warnings.warn(f"Value msut be between 0 and 1 but is {value:.3f}. Clipping value to 1..")
            value = 1.
        elif value < 0.:
            warnings.warn(f"Value msut be between 0 and 1 but is {value:.3f}. Clipping value to 0..")
            value = 0.
        point = self.interpolate(value, normalized=True)
        return point.x, point.y
    
    def closest_from(self, x, y) -> tuple[int, tuple]:
        min_dist = float('inf')
        wpt_closest = None
        for idx, wpt in enumerate(self._waypoints):
            wx, wy = wpt
            d = (wx - x)**2 + (wy - y)**2
            if d < min_dist:
                idx_closest = idx
                wpt_closest = wpt
        
        return idx_closest, wpt_closest

    def wpt_closest_from(self, x, y):
        return self.closest_from(x, y)[1]

    def idx_closest_from(self, x, y):
        return self.closest_from(x, y)[0]
    
    def get_default_headings_deg(self, seacharts_frame:bool=True) -> list:
        """
        Compute headings based on x, y positions. 
        """
        headings_deg = []
        w_prev = self.waypoints[0]
        wpt = self.waypoints[1]
        heading_deg = compute_heading_deg_from_wpts(w_prev, wpt, seacharts_frame=seacharts_frame)
        headings_deg.append(heading_deg)                                     

        for i in range(1, len(self._waypoints)-1):
            heading_deg = compute_heading_deg_from_wpts(self._waypoints[i-1], self._waypoints[i+1], seacharts_frame=seacharts_frame)
            headings_deg.append(heading_deg)

        heading_deg = compute_heading_deg_from_wpts(self._waypoints[i], self._waypoints[i+1], seacharts_frame=seacharts_frame)
        headings_deg.append(heading_deg)
        return headings_deg
    
    @property
    def waypoints(self) -> list:
        return self._waypoints

    
class TimeStampedWaypoints(Waypoints):
    def __init__(self, timestamped_waypoints:list=None, dim_idx_for_viz:tuple=(0, 1)):
        """
        list of tuples structured as (t, obj) where object must have a __getitem__ method implemented
        """
        self._dim_idx_for_viz = dim_idx_for_viz
        self._timestamped_waypoints = timestamped_waypoints or []
        waypoints = [wpt[1] for wpt in self._timestamped_waypoints]
        self._times = [wpt[0] for wpt in self._timestamped_waypoints]

        if len(self) <= 0:
            warnings.warn(f"{self} object instantiated without any timestamped waypoint")
            self._t0, self._tf, self._wpt_type = None, None, None
        else:
            self._t0, self._tf, self._wpt_type = self._times[0], self._times[-1], type(waypoints[0])
        super().__init__(waypoints=[(wpt[dim_idx_for_viz[0]], wpt[dim_idx_for_viz[1]]) for wpt in waypoints])

    def __len__(self) -> int:
        return len(self._timestamped_waypoints)

    def __call__(self, t:float) -> Any:
        assert len(self) > 0, f"Impossible to call {self} object because no timestamped waypoints were provided beforehand"
        if t <= self._t0:
            return self._timestamped_waypoints[0][1]
        
        elif t >= self._tf:
            return self._timestamped_waypoints[-1][1]
        
        for i, point_i in enumerate(self._timestamped_waypoints):
            t_i, obj = point_i
            if t_i > t:
                t_prev, obj_prev = self._timestamped_waypoints[i-1]
                factor = (t-t_prev) / (t_i-t_prev)
                val_list = []
                for val, val_prev in zip(obj, obj_prev):
                    val_list.append(factor * (val-val_prev) + val_prev)
                if self._wpt_type == tuple:
                    return self._wpt_type(val_list)
                else:
                    return self._wpt_type(*val_list)
            
    def plot(self, t0:float, tf:float, N:int, *args, mode='traj', ax=None, text:bool=False, grid:bool=True, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        return self._plot_and_scatter_wrapper(ax.plot.__name__, t0, tf, N, ax, *args, mode=mode, text=text, grid=grid, **kwargs)
        
    def scatter(self, t0:float, tf:float, N:int, *args, mode='traj', ax=None, text:bool=False, grid:bool=True, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        return self._plot_and_scatter_wrapper(ax.scatter.__name__, t0, tf, N, ax, *args, mode=mode, text=text, grid=grid, **kwargs)

    def _plot_and_scatter_wrapper(self, func_name:str, t0:float, tf:float, N:int, ax, *args, mode='traj', text:bool=False, grid:bool=True, **kwargs):
        dt = (tf-t0)/N
        x = []
        y = []
        t = []
        for i in range(N):
            t_i = t0 + dt * i
            obj_i = self(t_i)
            x_i, y_i = obj_i[self._dim_idx_for_viz[0]], obj_i[self._dim_idx_for_viz[1]]
            x.append(x_i)
            y.append(y_i)
            t.append(t_i)

        if text:
            for xi, yi, ti in zip(x, y, t):
                ax.text(xi, yi, f"{ti:.2f}")

        if mode=='traj':
            getattr(ax, func_name)(x, y, *args, **kwargs) # call the func_name method of ax object
            ax.grid(grid)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        elif mode=='x':
            getattr(ax, func_name)(t, x, *args, **kwargs) # call the func_name method of ax object
            ax.grid(grid)
            ax.set_xlabel('t')
            ax.set_ylabel('x')
        elif mode=='y':
            getattr(ax, func_name)(t, y, *args, **kwargs) # call the func_name method of ax object
            ax.grid(grid)
            ax.set_xlabel('t')
            ax.set_ylabel('y')
        else:
            raise ValueError(f"mode must be either traj, x or y but is {mode}")
        return ax
    
    def get_times_in_sql_format(self, t0:str, format:str=TIME_FORMAT) -> list[datetime]:
        t0 = datetime.strptime(t0, format)
        times_sql = []
        for t in self._times:
            t_sql = t0 + timedelta(seconds=t)
            times_sql.append(t_sql)
        return times_sql
    
    def get_colors_from_time(self, colormap:str='viridis') -> list[str]:
        t_min = min(self._times)
        t_max = max(self._times)
        norm = mat_colors.Normalize(vmin=t_min, vmax=t_max)
        cmap = plt.get_cmap(colormap)
        
        colors = []
        for ti in self._times:
            rgb = cmap(norm(ti))[:3]
            color = mat_colors.rgb2hex(rgb)
            colors.append(color)

        return colors

    
    def to_sql(self, path_to_database:str, mmsi:str, colormap:str='viridis', table:str='AIS', t0:str='26-08-2024 08:00:00', heading_in_seacharts_frame:bool=True) -> None:
        """
        Save timestamped waypoints into a SQL database. 
        """
        headings = self.get_default_headings_deg(seacharts_frame=heading_in_seacharts_frame)
        times_sql = self.get_times_in_sql_format(t0)
        colors = self.get_colors_from_time(colormap=colormap)

        with sqlite3.connect(path_to_database) as connection:
            cursor = connection.cursor()

            # Get existing tables in the database
            list_of_tables = cursor.execute(f"""SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'; """).fetchall()
            
            # Check if table does already exist in database
            table_exist = False
            for table_i in list_of_tables:
                if table in table_i:
                    table_exist = True

            # If table does not exist, create it
            if not table_exist:
                cursor.execute(f"""CREATE TABLE {table} (mmsi text, lon int, lat int, heading float, last_updated text, color text)""")

            for wpt_i, heading_i, time_sql_i, color_i in zip(self._waypoints, headings, times_sql, colors):
                cursor.execute(f"""INSERT INTO {table} (mmsi, lon, lat, heading, last_updated, color) VALUES ('{mmsi}', {wpt_i[0]}, {wpt_i[1]}, {heading_i}, '{time_sql_i.strftime(format=TIME_FORMAT)}', '{color_i}')""")

                
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{type(self).__name__}"

def compute_heading_deg_from_wpts(wpt1, wpt2, seacharts_frame:bool=True) -> float:
    x1, y1 = wpt1
    x2, y2 = wpt2
    angle = 180 * atan2((x1-x2), (y2-y1)) / pi
    return -angle if seacharts_frame else angle

def test():
    import numpy as np
    waypoints = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    obstacle = Obstacle([(0, 0), (1, 0), (1, 1), (0, 1)])
    path = Waypoints(waypoints)
    interpolated_path = path.resample(10)
    print(path(0.1))
    ax = interpolated_path.plot()
    interpolated_path.scatter(ax=ax)
    obstacle.plot(ax=ax)
    plt.show()

    wpts_traj = TimeStampedWaypoints([(0, (0, -1)), (1, (0.5, -2)), (2, (1, 1)), (3, (2, 0.5)), (4, (1.5, 0))])
    ax = wpts_traj.plot(-8, 12, 100, mode='y')
    wpts_traj.scatter(-8, 12, 100, ax=ax, mode='x')
    plt.show()

    ax = wpts_traj.plot(-8, 12, 100, mode='traj', text=True)
    wpts_traj.scatter(-8, 12, 100, mode='traj', ax=ax)
    plt.show()

    print(wpts_traj(0.3))

    empty_traj = TimeStampedWaypoints() # Must raise a user warning
    empty_traj(10) # Must raise an assertion error (No waypoints were provided for interpolation)


def test_db():
    import numpy as np

    wpts_traj = TimeStampedWaypoints([(0, (0, -1)), (1, (0.5, -2)), (2, (1, 1)), (3, (2, 0.5)), (4, (1.5, 0))])
    ax = wpts_traj.plot(-8, 12, 100, mode='traj', text=True)
    wpts_traj.scatter(-8, 12, 100, mode='traj', ax=ax)
    wpts_traj.to_sql('test_db.db', '111111111')
    ax.axis('equal')
    plt.show()





if __name__ == "__main__":
    # test()
    test_db()