from shapely import LineString
from nav_env.geometry.wrapper import GeometryWrapper
from nav_env.obstacles.obstacles import Obstacle
import matplotlib.pyplot as plt, matplotlib.colors as mat_colors
import warnings, sqlite3, casadi as cd, numpy as np
from datetime import datetime, timedelta
from math import atan2, pi
from typing import Any, Callable
from seacharts import ENC
TIME_FORMAT = "%d-%m-%Y %H:%M:%S"

class Waypoints(GeometryWrapper):
    def __init__(self, waypoints: list = None):
        self._waypoints = [] if waypoints is None else waypoints
        super().__init__(xy=self._waypoints, geometry_type=LineString)
        self._alphas = self.get_alphas()
        self._segment_fn = self.get_segment_fn()
        self._segments = self.get_segment()
        # print([0] + self._alphas)
        # print(np.array(waypoints, dtype=int).T.shape)
        # print([[w[0] for w in waypoints], [w[1] for w in waypoints]], [0.0] + self._alphas)
        self._interp_x = cd.interpolant('rx', 'bspline', [[0.0] + self._alphas], [w[0] for w in waypoints] )
        self._interp_y = cd.interpolant('ry', 'bspline', [[0.0] + self._alphas], [w[1] for w in waypoints] )
        
    def get_alphas(self) -> tuple:
        d = []
        for i, wpt_i in enumerate(self._waypoints):
            if i>0:  
                d_i = ((wpt_i[0]-wpt_prev[0])**2 + (wpt_i[1]-wpt_prev[1])**2)**0.5
                if i==1:
                    d.append(d_i)
                elif i>1:
                    d.append(d_i + d[-1])
            wpt_prev = wpt_i
        alpha = []
        for d_i in d:
            alpha.append(d_i/d[-1])
        self.d_tot = d[-1]
        return alpha
    
    def get_segment_fn(self) -> tuple[Callable]:
        """
        All segment_fn returns a point that belong to the infinite line defined by previous and next waypoint
        However the valid segment are only obtained when the value of alpha matches the right segment.        
        """
        func = []
        for i, wpt_i in enumerate(self._waypoints):
            if i > 0:
                func_i = lambda alpha, big_alpha=self._alphas[i-1], i=i, wpt_prev=wpt_prev, wpt_i=wpt_i: cd.vertcat(
                    wpt_prev[0] * (1-alpha/big_alpha) + alpha * wpt_i[0]/big_alpha,
                    wpt_prev[1] * (1-alpha/big_alpha) + alpha * wpt_i[1]/big_alpha
                )
                func.append(func_i)
            wpt_prev = wpt_i
        return func
    
    def get_segment(self) -> list[tuple[float, float, Callable]]:
        """
        List of tuples containing (alpha_start, alpha_end, segment_fn) for each segment
        """
        new_alphas = [0.] + list(self._alphas)
        segments = []
        for i in range(len(new_alphas)-1):
            alpha_start = new_alphas[i]
            alpha_end = new_alphas[i+1]
            segment_fn = self._segment_fn[i]
            segments.append((alpha_start, alpha_end, segment_fn))
        return segments
    
    def r_old(self, alpha:cd.SX) -> Any:
        """
        (x, y) = r(alpha) --> ONLY TO BE USED WITH CASADI, OTHERWISE SIMPLY USE __call__ which uses shapely's API
        """
        # segments: list of (alpha_start, alpha_end, segment_fn)
        result = cd.vertcat(0,0)
        for alpha_start, alpha_end, segment_fn in self._segments:
            # Indicator: 1 if alpha in [alpha_start, alpha_end), else 0
            in_segment = cd.logic_and(alpha >= alpha_start, alpha < alpha_end)
            # Use if_else to select the segment function output or 0
            result += cd.if_else(in_segment, segment_fn(alpha), 0)
        return result
    
    def r(self, alpha:cd.SX) -> Any:
        return cd.vertcat(self._interp_x(alpha), self._interp_y(alpha))
    
    def test_r(self, alpha:cd.SX) -> Any:
        """
        (x, y) = r(alpha) --> ONLY TO BE USED WITH CASADI, OTHERWISE SIMPLY USE __call__ which uses shapely's API
        """
        # segments: list of (alpha_start, alpha_end, segment_fn)
        result = 0
        for alpha_start, alpha_end, segment_fn in self._segments:
            # Indicator: 1 if alpha in [alpha_start, alpha_end), else 0
            in_segment = cd.logic_and(alpha >= alpha_start, alpha < alpha_end)
            # Use if_else to select the segment function output or 0
            result += cd.if_else(in_segment, segment_fn(alpha), 0)
        return result

    def __iter__(self):
        return iter(self.get_xy_as_list())
    
    def __getitem__(self, idx):
        xy = self.get_xy_as_list()
        assert idx < len(xy), "Index out of range."
        return xy[idx]
    
    def __call__(self, value:float) -> tuple[float, float]:
        # if value > 1.:
        #     warnings.warn(f"Value msut be between 0 and 1 but is {value:.3f}. Clipping value to 1..")
        #     value = 1.
        # elif value < 0.:
        #     warnings.warn(f"Value msut be between 0 and 1 but is {value:.3f}. Clipping value to 0..")
        #     value = 0.
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

    @staticmethod
    def from_trajectory_fn(traj_fn:Callable, timestamps:list) -> "TimeStampedWaypoints":
        """
        Build a TimeStampedWaypoints object from a trajectory function a list of timestamps to specify
        when to evaluate the function.
        """
        return TimeStampedWaypoints([(ti, traj_fn(ti)) for ti in timestamps])

    def resample(self, N:int) -> "TimeStampedWaypoints":
        dt = (self._tf-self._t0)/(N-1)
        return TimeStampedWaypoints(
            timestamped_waypoints=[(n*dt+self._t0, self(n*dt+self._t0)) for n in range(0, N, 1)],
            dim_idx_for_viz=self._dim_idx_for_viz)

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
            # print(f"wpt{i}: ", t_i, obj)
            if t_i > t:
                # print("t_i > t")
                t_prev, obj_prev = self._timestamped_waypoints[i-1]
                factor = (t-t_prev) / (t_i-t_prev)
                # print("artificial t: ", factor * (t_i - t_prev) + t_prev)
                val_list = []
                for val, val_prev in zip(obj, obj_prev):
                    val_list.append(factor * (val-val_prev) + val_prev)
                if self._wpt_type == tuple:
                    return self._wpt_type(val_list)
                else:
                    return self._wpt_type(*val_list)
            
    def plot(self, *args, t0:float=None, tf:float=None, N:int=None, mode='traj', ax=None, text:bool=False, grid:bool=False, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        if t0 is None:
            t0 = self._t0
        if tf is None:
            tf = self._tf
        if N is None:
            N = len(self._timestamped_waypoints)
        return self._plot_and_scatter_wrapper(ax.plot.__name__, t0, tf, N, ax, *args, mode=mode, text=text, grid=grid, **kwargs)
        
    def scatter(self, t0:float=None, tf:float=None, N:int=None, *args, mode='traj', ax=None, text:bool=False, grid:bool=False, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        if t0 is None:
            t0 = self._t0
        if tf is None:
            tf = self._tf
        if N is None:
            N = len(self._timestamped_waypoints)
        return self._plot_and_scatter_wrapper(ax.scatter.__name__, t0, tf, N, ax, *args, mode=mode, text=text, grid=grid, **kwargs)

    def _plot_and_scatter_wrapper(self, func_name:str, t0:float, tf:float, N:int, ax, *args, mode='traj', text:bool=False, grid:bool=False, **kwargs):
        dt = (tf-t0)/N
        x = []
        y = []
        t = []
        v = []
        eps = 1e-6 * dt # small time increment to approximate derivative
        for i in range(N+1):
            t_i = t0 + dt * i
            obj_i = self(t_i)
            x_i, y_i = obj_i[self._dim_idx_for_viz[0]], obj_i[self._dim_idx_for_viz[1]]
            obj_eps = self(t_i + eps)
            x_eps, y_eps = obj_eps[self._dim_idx_for_viz[0]], obj_i[self._dim_idx_for_viz[1]]
            v_i = ((x_eps-x_i)**2 + (y_eps-y_i)**2)**0.5 / eps
            x.append(x_i)
            y.append(y_i)
            t.append(t_i)
            v.append(v_i)

        if text:
            for xi, yi, ti in zip(x, y, t):
                ax.text(xi, yi, f"{ti:.2f}")


        if mode=='speed_profile':
            getattr(ax, func_name)(t, v, *args, **kwargs) # call the func_name method of ax object
            ax.grid(grid)
            ax.set_xlabel('t')
            ax.set_ylabel('v')
        elif mode=='traj':
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
    
    def get_times_in_sql_format(self, t0:str, timestamps:list=None, format:str=TIME_FORMAT) -> list[datetime]:
        timestamps = self._times if timestamps is None else timestamps
        t0 = datetime.strptime(t0, format)
        times_sql = []
        for t in timestamps:
            t_sql = t0 + timedelta(seconds=t)
            times_sql.append(t_sql)
        return times_sql
    
    def get_colors_from_time(self, timestamps:list=None, colormap:str='viridis', alpha=1.) -> list[str]:
        timestamps = self._times if timestamps is None else timestamps
        t_min = min(self._times)
        t_max = max(self._times)
        norm = mat_colors.Normalize(vmin=t_min, vmax=t_max)
        cmap = plt.get_cmap(colormap)
        
        colors = []
        for ti in timestamps:
            rgb = cmap(norm(ti))[:3]
            # color = mat_colors.rgb2hex(rgb)
            color = mat_colors.to_hex((*rgb, alpha), keep_alpha=True)
            colors.append(color)

        return colors
    
    def plot_to_enc(self, enc:ENC, colormap:str='viridis', alpha=1., width:float=None, thickness:float=None, edge_style:str | tuple=None, marker_type:str=None) -> None:
        # Compute colors
        t_min = min(self._times)
        t_max = max(self._times)
        norm = mat_colors.Normalize(vmin=t_min, vmax=t_max)
        cmap = plt.get_cmap(colormap)

        t_prev = None
        wpt_prev = None
        for t, wpt in self._timestamped_waypoints:
            if t_prev is None:
                t_prev = t
                wpt_prev = wpt
                continue
            # Get corresponding color
            t_mean = (t+t_prev)/2
            rgb = cmap(norm(t_mean))[:3]
            color = mat_colors.to_hex((*rgb, alpha), keep_alpha=True)
            enc.display.draw_line([wpt_prev, wpt], color=color, width=width, thickness=thickness, edge_style=edge_style, marker_type=marker_type)
            t_prev = t
            wpt_prev = wpt
        return None
    
    def get_waypoints(self, timestamps:list=None) -> list:
        timestamps = self._times if timestamps is None else timestamps
        waypoints = []
        for ti in timestamps:
            waypoints.append(self(ti))
        return waypoints

    def to_sql(self,
               path_to_database:str,
               mmsi:int,
               timestamps:list=None,
               colormap:str='viridis',
               alpha:float=1.,
               table:str='AisHistory',
               t0:str='26-08-2024 08:00:00',
               heading_in_seacharts_frame:bool=True,
               clear_table:bool=False,
               length:float=None,
               width:float=None,
               scale:float=1.,
               isolate_timestamps:bool=False
               ) -> None:
        """
        Save timestamped waypoints into a SQL database. Currently, we use a trick to visualize the same ship multiple times. If we have to show one ship at N different timestamps,
        we create N different mmsi. If the input mmsi=100000000, and we have 3 timestamps to show, then we will have mmsi = [100000000, 100000001, 100000002]
        """
        if timestamps is not None and not isinstance(timestamps, list):
            timestamps = [timestamps]
        if isolate_timestamps:
            timestamps = [ti for ti in range(min(timestamps), max(timestamps)+1)]
            timestamps_for_sql = [ti*3600 for ti in timestamps]
            # timestamps_for_sql = []
            # for i, ti in enumerate(timestamps):
            #     new_ti = 3600 * ti
            #     timestamps_for_sql.append(new_ti)
        else:
            timestamps_for_sql = timestamps
        

        headings = self.get_default_headings_from_fn_deg(timestamps=timestamps, seacharts_frame=heading_in_seacharts_frame)
        times_sql = self.get_times_in_sql_format(t0, timestamps=timestamps_for_sql)
        colors = self.get_colors_from_time(timestamps=timestamps, colormap=colormap, alpha=alpha)
        waypoints = self.get_waypoints(timestamps=timestamps)

        with sqlite3.connect(path_to_database) as connection:
            cursor = connection.cursor()

            # Get existing tables in the database
            list_of_tables = cursor.execute(f"""SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'; """).fetchall()
            
            # Check if table does already exist in database
            table_exist = False
            for table_i in list_of_tables:
                if table in table_i:
                    table_exist = True

            # Clear table if it does exist and if it is required
            if table_exist and clear_table:
                cursor.execute(f"""DROP TABLE {table}""")
                table_exist = False

            # If table does not exist, create it
            if not table_exist:
                cursor.execute(f"""CREATE TABLE {table} (mmsi text, lon int, lat int, heading float, last_updated text, color text, length float, width float)""")

            for i, (wpt_i, heading_i, time_sql_i, color_i) in enumerate(zip(waypoints, headings, times_sql, colors)):
                cursor.execute(f"""INSERT INTO {table} (mmsi, lon, lat, heading, last_updated, color, length, width) VALUES ('{str(mmsi+i)}', {wpt_i[0]}, {wpt_i[1]}, {heading_i}, '{time_sql_i.strftime(format=TIME_FORMAT)}', '{color_i}', {length*scale if length is not None else "NULL"}, {width*scale if width is not None else "NULL"})""")

                
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{type(self).__name__}"
    
    def get_single_default_heading_from_fn_deg(self, t:float, eps:float=1e-6, seacharts_frame:bool=True) -> float:
        if t + eps > self._tf:
            wpt1 = self(self._tf-eps)
            wpt2 = self(self._tf)
        elif t - eps < self._t0:
            wpt1 = self(self._t0)
            wpt2 = self(self._t0 + eps)
        else:
            wpt1 = self(t-eps)
            wpt2 = self(t+eps)
        return compute_heading_deg_from_wpts(wpt1, wpt2, seacharts_frame=seacharts_frame)
    
    def get_default_headings_from_fn_deg(self, timestamps:list=None, eps:float=1e-6, seacharts_frame:bool=True) -> list[float]:
        timestamps = self._times if timestamps is None else timestamps
        headings = []
        for ti in timestamps:
            headings.append(self.get_single_default_heading_from_fn_deg(ti, eps=eps, seacharts_frame=seacharts_frame))
        return headings

def compute_heading_deg_from_wpts(wpt1, wpt2, seacharts_frame:bool=True) -> float:
    x1, y1 = wpt1[0], wpt1[1]
    x2, y2 = wpt2[0], wpt2[1]
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
    wpts_traj.to_sql('test_db.db', 100000000)
    ax.axis('equal')
    plt.show()





if __name__ == "__main__":
    # test()
    test_db()