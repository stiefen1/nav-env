from dataclasses import dataclass
import json, os, pathlib, numpy as np
from nav_env.wind.wind_vector import WindVector
from nav_env.water.water_vector import WaterVector
from nav_env.control.command import GeneralizedForces
from nav_env.ships.states import ShipStates3, ShipTimeDerivatives3

PATH_TO_DEFAULT_JSON = os.path.join(pathlib.Path(__file__).parent, "default_ship_params.json")
PATH_TO_DEFAULT_NEW_JSON = os.path.join(pathlib.Path(__file__).parent, "new_ship_params.json")

@dataclass
class ShipPhysicalParams:
    """
    Base class for ship parameters.
    """
    help: dict
    inertia: dict
    mass_over_linear_friction_coefficient: dict
    nonlinear_friction_coefficient: dict
    added_mass_coefficient: dict
    dimensions: dict
    wind: dict
    water: dict

    @staticmethod
    def load_from_json(json_file):
        with open(json_file, "r") as f:
            params = json.load(f)
        return ShipPhysicalParams(**params)
    
    @staticmethod
    def default():
        return ShipPhysicalParams.load_from_json(PATH_TO_DEFAULT_NEW_JSON)


class ShipPhysics:
    def __init__(self, path_to_physical_params:str=PATH_TO_DEFAULT_NEW_JSON):
        self._params = ShipPhysicalParams.load_from_json(path_to_physical_params)
        self._mass, self._x_g = self._params.inertia['mass'], self._params.inertia['x_g']
        self._i_z = self.__get_moment_of_inertia_about_z()
        self._x_du, self._y_dv, self._n_dr = self.__get_added_mass()
        self._length, self._width, self._proj_area_front, self._proj_area_side = self.__get_dimensions_and_projection()
        self._t_surge, self._t_sway, self._t_yaw = self.__get_mass_over_linear_friction_coefficients()
        self._ku, self._kv, self._kr = self.__get_nonlinear_friction_coefficients()
        self._rho_a, self._cx, self._cy, self._cn = self.__get_wind_coefficients()
        self._coriolis_matrix_fn = self.__get_coriolis_matrix_fn()
        self._inv_mass_matrix = self.__get_inv_mass_matrix()
        self._coriolis_added_matrix_fn = self.__get_coriolis_added_matrix_fn()
        self._linear_damping_matrix = self.__get_linear_damping_matrix()
        self._nonlinear_damping_matrix_fn = self.__get_nonlinear_damping_matrix_fn()
        self._rotation_matrix_fn = self.__get_rotation_matrix_fn()

    def __get_moment_of_inertia_about_z(self):
        l = self._params.dimensions['length']
        w = self._params.dimensions['width']
        return  self._mass * (l**2 + w**2) / 12
    
    def __get_added_mass(self):
        x_du = self._mass * self._params.added_mass_coefficient['surge']
        y_dv = self._mass * self._params.added_mass_coefficient['sway']
        n_dr = self._i_z * self._params.added_mass_coefficient['yaw']
        return x_du, y_dv, n_dr
    
    def __get_dimensions_and_projection(self):
        dimensions = self._params.dimensions
        proj_area_front = dimensions['width'] * dimensions['h_front']
        proj_area_side = dimensions['length'] * dimensions['h_side']
        return dimensions['length'], dimensions['width'], proj_area_front, proj_area_side

    def __get_nonlinear_friction_coefficients(self):
        nonlinear_friction = self._params.nonlinear_friction_coefficient
        return nonlinear_friction['surge'], nonlinear_friction['sway'], nonlinear_friction['yaw']
    
    def __get_wind_coefficients(self):
        wind = self._params.wind
        return wind['rho_a'], wind['cx'], wind['cy'], wind['cn']

    def __get_mass_over_linear_friction_coefficients(self):
        mass_over_linear_friction_coefficient = self._params.mass_over_linear_friction_coefficient
        return mass_over_linear_friction_coefficient['surge'], mass_over_linear_friction_coefficient['sway'], mass_over_linear_friction_coefficient['yaw']  

    def __get_inv_mass_matrix(self):
        return np.linalg.inv(np.array([
                [self._mass + self._x_du, 0., 0.],
                [0., self._mass + self._y_dv, self._mass * self._x_g],
                [0., self._mass * self._x_g, self._i_z + self._n_dr]
            ]))

    def __get_coriolis_matrix_fn(self):
        def coriolis_matrix_fn(x_dot, y_dot, psi_dot_rad):
            return np.array([
                [0., 0., -self._mass * (self._x_g * psi_dot_rad + y_dot)],
                [0., 0., self._mass * x_dot],
                [self._mass * (self._x_g * psi_dot_rad + y_dot), -self._mass * x_dot, 0.]
            ])
        return coriolis_matrix_fn

    def __get_coriolis_added_matrix_fn(self):
        def coriolis_added_matrix_fn(u_r, v_r):
            return np.array([
                [0., 0., self._y_dv * v_r],
                [0., 0., -self._x_du * u_r],
                [-self._y_dv * v_r, self._x_du * u_r, 0.]
            ])
        return coriolis_added_matrix_fn
    
    def __get_linear_damping_matrix(self):
        return np.array([
            [self._mass / self._t_surge, 0., 0.],
            [0., self._mass / self._t_sway, 0.],
            [0., 0., self._i_z / self._t_yaw]
        ])
    
    def __get_nonlinear_damping_matrix_fn(self):
        def nonlinear_damping_matrix_fn(x_dot, y_dot, psi_dot_rad):
            return np.array([
                [self._ku * np.abs(x_dot), 0., 0.],
                [0., self._kv * np.abs(y_dot), 0.],
                [0., 0., self._kr * np.abs(psi_dot_rad)]
            ])
        return nonlinear_damping_matrix_fn
    
    def __get_rotation_matrix_fn(self):
        def rotation_matrix_fn(psi_rad, dim:int=3):
            # psi_rad -= np.pi/2
            return np.array([
                [-np.sin(psi_rad), np.cos(psi_rad), 0.],
                [np.cos(psi_rad), np.sin(psi_rad), 0.],
                [0., 0., -1.]
            ])[0:dim, 0:dim] # Transpose or not, it doesn't change anything!!! This transformation is correct!
        return rotation_matrix_fn
    
    def get_wind_force(self, wind:WindVector, x_dot:float, y_dot:float, yaw:float) -> GeneralizedForces:
        """
        Get the wind force acting on the ship, in the ship frame.
        """
        beta_w = -wind.direction # Wind direction is given in the world frame
        psi = -yaw # Yaw angle is given in the world frame

        # Compute wind speed in ship frame
        vx_in_ship = wind.speed * np.cos(psi-beta_w)
        vy_in_ship = -wind.speed * np.sin(psi-beta_w)

        # Compute relative speed in ship frame
        u_rw = vx_in_ship - x_dot
        v_rw = vy_in_ship - y_dot

        # gamma_rw = -np.arctan2(v_rw, u_rw) # Wind direction w.r.t yaw angle in ship frame
        gamma_rw = psi - beta_w - np.pi
        wind_rw2 = u_rw ** 2 + v_rw ** 2
        c_x = -self._cx * np.cos(gamma_rw) # TODO: Check if this is correct, originally it was c_x = -self._cx * np.cos(gamma_rw)
        c_y = self._cy * np.sin(gamma_rw)
        c_n = self._cn * np.sin(2 * gamma_rw)
        # print(f"beta_w: {beta_w*180/np.pi:.2f} | psi: {psi*180/np.pi:.2f} | vx_in_ship: {vx_in_ship:.2f}, vy_in_ship: {vy_in_ship:.2f} | v_rw: {v_rw:.2f}, u_rw: {u_rw:.2f} | gamma_rw: {gamma_rw*180/np.pi:.2f} | wind_rw2: {wind_rw2:.2f} | c_x: {c_x:.2f}, c_y: {c_y:.2f}, c_n: {c_n:.2f}")
        # print(f"wind: {(wind.direction)*180/np.pi:.2f} | angle: {angle*180/np.pi:.2f} | yaw: {yaw*180/np.pi:.2f} | uw: {uw:.2f}, vw: {vw:.2f} | u_rw: {u_rw:.2f}, v_rw: {v_rw:.2f} | gamma_rw: {gamma_rw*180/np.pi:.2f} | wind_rw2: {wind_rw2:.2f} | c_x: {c_x:.2f}, c_y: {c_y:.2f}, c_n: {c_n:.2f}")
        tau_coeff = 0.5 * self._rho_a * wind_rw2
        tau_u = tau_coeff * c_x * self._proj_area_front
        tau_v = tau_coeff * c_y * self._proj_area_side
        tau_n = tau_coeff * c_n * self._proj_area_side * self._length
        # print(f"w: {wind.direction:.2f}, {wind.speed:.2f} | xy_dot: {x_dot:.2f}, {y_dot:.2f} | yaw: {yaw:.2f} | uv_rw: {u_rw:.2f}, {v_rw:.2f} | tau: {tau_u:.2f}, {tau_v:.2f}, {tau_n:.2f}")
        return GeneralizedForces(float(tau_u), float(tau_v), 0., 0., 0., float(tau_n))
    
    def get_water_force(self, water:WaterVector, x_dot:float, y_dot:float, yaw:float) -> GeneralizedForces:
        """
        Get the water force acting on the ship.
        """
        return GeneralizedForces()
    
    def get_time_derivatives(self, states:ShipStates3, wind:WindVector=WindVector((0., 0.), vector=(0., 0.)), water:WaterVector=WaterVector((0., 0.), vector=(0., 0.)), control_forces:GeneralizedForces=GeneralizedForces(), external_forces:GeneralizedForces=GeneralizedForces()) -> ShipTimeDerivatives3:
        """
        All inputs are in the world frame.
        """
        R2d = self._rotation_matrix_fn(states.psi_rad, dim=2)
        R3d = self._rotation_matrix_fn(states.psi_rad, dim=3)
        pose_dot_in_ship_frame = np.dot(R3d, states.vel)

        states_in_ship_frame = ShipStates3(0., 0., 0., *pose_dot_in_ship_frame)
        # print(states_in_ship_frame)

        # Compute external forces acting on the ship (wind, water, control, external)
        wind_force = self.get_wind_force(wind, states_in_ship_frame.x_dot, states_in_ship_frame.y_dot, states.psi_rad)
        water_force = self.get_water_force(water, states_in_ship_frame.x_dot, states_in_ship_frame.y_dot, states.psi_rad)
        # print(water_force)
        total_force_in_ship_frame:GeneralizedForces = wind_force + water_force + control_forces + external_forces # There is a wind force always acting on the ship
        # print(wind_force.f_y, water_force.f_y, control_forces.f_y, external_forces.f_y, total_force.f_y)

        # Transform water current velocity to ship frame
        current_vel_in_ship_frame = R2d @ water.velocity
        u_r = states_in_ship_frame.x_dot - current_vel_in_ship_frame[0] # relative speed in surge w.r.t water
        v_r = states_in_ship_frame.y_dot - current_vel_in_ship_frame[1] # relative speed in sway w.r.t water
        vel_relative_to_current_in_ship_frame = np.array([u_r, v_r, states_in_ship_frame.psi_dot_rad])
        # print(f"u_r: {u_r:.2f}, v_r: {v_r:.2f}")

        # Compute matrices linked to the ship dynamics
        Crb = self._coriolis_matrix_fn(states_in_ship_frame.x_dot, states_in_ship_frame.y_dot, states_in_ship_frame.psi_dot_rad)
        Ca = self._coriolis_added_matrix_fn(u_r, v_r)
        Dl = self._linear_damping_matrix
        Dnl = self._nonlinear_damping_matrix_fn(*vel_relative_to_current_in_ship_frame)

        # Compute the time derivatives of the states
        # acc = np.dot(
        #     self._inv_mass_matrix,
        #     - np.dot(Crb, states_in_ship_frame.vel)
        #     - np.dot(Ca, vel_relative_to_current)
        #     - np.dot(Dl + Dnl, vel_relative_to_current)
        #     + total_force.uvn
        # )

        acc1 = -self._inv_mass_matrix @ (Crb @ states_in_ship_frame.vel)
        acc2 = -self._inv_mass_matrix @ (Ca @ vel_relative_to_current_in_ship_frame)
        acc3 = -self._inv_mass_matrix @ (Dl @ vel_relative_to_current_in_ship_frame)
        acc4 = -self._inv_mass_matrix @ (Dnl @ vel_relative_to_current_in_ship_frame)
        acc5 = self._inv_mass_matrix @ total_force_in_ship_frame.uvn
        acc = acc1 + acc2 + acc3 + acc4 + acc5

        idx = 0
        # print((R3d @ acc5)[idx])
        # print((R3d @ acc1)[idx], (R3d @ acc2)[idx], (R3d @ acc3)[idx], (R3d @ acc4)[idx], (R3d @ acc5)[idx], (R3d @ acc)[idx])
        # print(f"acc1: {(R3d @ acc1)[idx]:.2f}, acc2: {(R3d @ acc2)[idx]:.2f}, acc3: {(R3d @ acc3)[idx]:.2f}, acc4: {(R3d @ acc4)[idx]:.2f}, acc5: {(R3d @ acc5)[idx]:.2f}, acc: {(R3d @ acc)[idx]:.2f}")

        # Transform the acceleration back to the world frame
        acc_in_world_frame = R3d @ acc
        # print(acc_in_world_frame)

        # TODO: Make the resulting forces plottable
        return ShipTimeDerivatives3(states.x_dot, states.y_dot, states.psi_dot_deg, float(acc_in_world_frame[0]), float(acc_in_world_frame[1]), float(acc_in_world_frame[2]))

    def __repr__(self):
        return f"{type(self).__name__} Object"

    @property
    def inv_mass_matrix(self):
        return self._inv_mass_matrix

    @property
    def coriolis_matrix(self):
        return self._coriolis_matrix_fn
    
    @property
    def coriolis_added_mass_matrix(self):
        return self._coriolis_added_matrix_fn
    
    @property
    def linear_damping_matrix(self):
        return self._linear_damping_matrix

    @property
    def nonlinear_damping_matrix(self):
        return self._nonlinear_damping_matrix_fn
    
    @property
    def rotation_matrix(self):
        return self._rotation_matrix_fn

    @property
    def i_z(self):
        return self._i_z
    
    @property
    def mass(self):
        return self._mass
    
    @property
    def x_du(self):
        return self._x_du
    
    @property
    def y_dv(self):
        return self._y_dv
    
    @property
    def n_dr(self):
        return self._n_dr
    
    @property
    def t_surge(self):
        return self._t_surge
    
    @property
    def t_sway(self):
        return self._t_sway
    
    @property
    def t_yaw(self):
        return self._t_yaw
    
    @property
    def ku(self):
        return self._ku
    
    @property
    def kv(self):
        return self._kv
    
    @property
    def kr(self):
        return self._kr
    
    @property
    def params(self):
        return self._params
    
    @property
    def x_g(self):
        return self._x_g
    
    @property
    def length(self):
        return self._length
    
    @property
    def width(self):
        return self._width
    
    @property
    def proj_area_front(self):
        return self._proj_area_front
    
    @property
    def proj_area_side(self):
        return self._proj_area_side
    
    @property
    def rho_a(self):
        return self._rho_a


def test():
    # Load the ship parameters from json file

    ship_physics = ShipPhysics()
    print(ship_physics)
    print(ship_physics.inv_mass_matrix)
    print(ship_physics.coriolis_matrix(1, 2, 3))
    print(ship_physics.coriolis_added_mass_matrix(1, 2))
    print(ship_physics.linear_damping_matrix)
    print(ship_physics.nonlinear_damping_matrix(1, 2, 3))

    state = ShipStates3(1., 2., 3., 4., 5., 6.)
    wind = WindVector((0., 0.), vector=(1., 0.))

    print(ship_physics.get_time_derivatives(state, wind))



if __name__ == "__main__":
    test()