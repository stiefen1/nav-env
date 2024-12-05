from nav_env.ships.states import ShipStates3, ShipTimeDerivatives3
from nav_env.control.command import GeneralizedForces
from nav_env.wind.wind_vector import WindVector
from nav_env.water.water_vector import WaterVector
from nav_env.ships.params import ShipPhysicalParams, PATH_TO_DEFAULT_NEW_JSON
import numpy as np
from typing import Callable

class ShipPhysics:
    def __init__(self, path_to_physical_params:str=PATH_TO_DEFAULT_NEW_JSON) -> None:
        self._params = ShipPhysicalParams.load_from_json(path_to_physical_params)
        self._mass, self._x_g = self._params.inertia['mass'], self._params.inertia['x_g']
        self._i_z = self.__get_moment_of_inertia_about_z()
        self._x_du, self._y_dv, self._n_dr = self.__get_added_mass()
        self._length, self._width, self._proj_area_front, self._proj_area_side = self.__get_dimensions_and_projection()
        self._t_surge, self._t_sway, self._t_yaw = self.__get_mass_over_linear_friction_coefficients()
        self._ku, self._kv, self._kr = self.__get_nonlinear_friction_coefficients()
        self._rho_a, self._cx, self._cy, self._cn = self.__get_wind_coefficients()
        self._inv_mass_matrix = self.__get_inv_mass_matrix()
        self._linear_damping_matrix = self.__get_linear_damping_matrix()

    def __get_moment_of_inertia_about_z(self) -> float:
        l = self._params.dimensions['length']
        w = self._params.dimensions['width']
        return  self._mass * (l**2 + w**2) / 12
    
    def __get_added_mass(self) -> tuple[float, float, float]:
        x_du = self._mass * self._params.added_mass_coefficient['surge']
        y_dv = self._mass * self._params.added_mass_coefficient['sway']
        n_dr = self._i_z * self._params.added_mass_coefficient['yaw']
        return x_du, y_dv, n_dr
    
    def __get_dimensions_and_projection(self) -> tuple[float, float, float, float]:
        dimensions = self._params.dimensions
        proj_area_front = dimensions['width'] * dimensions['h_front']
        proj_area_side = dimensions['length'] * dimensions['h_side']
        return dimensions['length'], dimensions['width'], proj_area_front, proj_area_side

    def __get_nonlinear_friction_coefficients(self) -> tuple[float, float, float]:
        nonlinear_friction = self._params.nonlinear_friction_coefficient
        return nonlinear_friction['surge'], nonlinear_friction['sway'], nonlinear_friction['yaw']
    
    def __get_wind_coefficients(self) -> tuple[float, float, float, float]:
        wind = self._params.wind
        return wind['rho_a'], wind['cx'], wind['cy'], wind['cn']

    def __get_mass_over_linear_friction_coefficients(self) -> tuple[float, float, float]:
        mass_over_linear_friction_coefficient = self._params.mass_over_linear_friction_coefficient
        return mass_over_linear_friction_coefficient['surge'], mass_over_linear_friction_coefficient['sway'], mass_over_linear_friction_coefficient['yaw']  

    def __get_inv_mass_matrix(self) -> np.ndarray:
        return np.linalg.inv(np.array([
                [self._mass + self._x_du, 0., 0.],
                [0., self._mass + self._y_dv, self._mass * self._x_g],
                [0., self._mass * self._x_g, self._i_z + self._n_dr]
            ]))

    def __get_coriolis_matrix(self, x_dot, y_dot, psi_dot_rad) -> np.ndarray:
        return np.array([
                [0., 0., -self._mass * (self._x_g * psi_dot_rad + y_dot)],
                [0., 0., self._mass * x_dot],
                [self._mass * (self._x_g * psi_dot_rad + y_dot), -self._mass * x_dot, 0.]
                ])

    def __get_coriolis_added_matrix(self, u_r, v_r) -> np.ndarray:
        return np.array([
            [0., 0., self._y_dv * v_r],
            [0., 0., -self._x_du * u_r],
            [-self._y_dv * v_r, self._x_du * u_r, 0.]
        ])
    
    def __get_linear_damping_matrix(self) -> np.ndarray:
        return np.array([
            [self._mass / self._t_surge, 0., 0.],
            [0., self._mass / self._t_sway, 0.],
            [0., 0., self._i_z / self._t_yaw]
        ])
    
    def __get_nonlinear_damping_matrix(self, x_dot, y_dot, psi_dot_rad) -> np.ndarray:
        return np.array([
            [self._ku * np.abs(x_dot), 0., 0.],
            [0., self._kv * np.abs(y_dot), 0.],
            [0., 0., self._kr * np.abs(psi_dot_rad)]
        ])
    
    def __get_rotation_matrix(self, psi_rad, dim:int=3) -> np.ndarray:
        return np.array([
            [-np.sin(psi_rad), np.cos(psi_rad), 0.],
            [np.cos(psi_rad), np.sin(psi_rad), 0.],
            [0., 0., -1.]
        ])[0:dim, 0:dim] # Transpose or not, it doesn't change anything!!! This transformation is correct!
    
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
    
    def get_time_derivatives_and_forces(self,
                             states:ShipStates3,
                             wind:WindVector=WindVector((0., 0.), vector=(0., 0.)),
                             water:WaterVector=WaterVector((0., 0.), vector=(0., 0.)),
                             control_forces:GeneralizedForces=GeneralizedForces(),
                             external_forces:GeneralizedForces=GeneralizedForces()
                             ) -> tuple[ShipTimeDerivatives3, GeneralizedForces]:
        """
        All inputs are in the world frame.
        """
        R2d = self.__get_rotation_matrix(states.psi_rad, dim=2)
        R3d = self.__get_rotation_matrix(states.psi_rad, dim=3)
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
        Crb = self.__get_coriolis_matrix(states_in_ship_frame.x_dot, states_in_ship_frame.y_dot, states_in_ship_frame.psi_dot_rad)
        Ca = self.__get_coriolis_added_matrix(u_r, v_r)
        Dl = self._linear_damping_matrix
        Dnl = self.__get_nonlinear_damping_matrix(*vel_relative_to_current_in_ship_frame)

        f1 = -(Crb @ states_in_ship_frame.vel)
        f2 = -(Ca @ vel_relative_to_current_in_ship_frame)
        f3 = -(Dl @ vel_relative_to_current_in_ship_frame)
        f4 = -(Dnl @ vel_relative_to_current_in_ship_frame)
        f5 = np.array(total_force_in_ship_frame.uvn)

        # idx = 0
        # print(f"Crb: {f1[idx]} | Ca: {f2[idx]} | Dl: {f3[idx]} | Dnl: {f4[idx]} | F: {f5[idx]}")

        # Sum all forces acting on the ship
        f = f1 + f2 + f3 + f4 + f5

        # Compute the acceleration in the ship frame
        acc = self._inv_mass_matrix @ f

        # Transform the acceleration / forces back to the world frame
        f_in_world = R3d @ f
        acc_in_world_frame = R3d @ acc

        # TODO: Make the resulting forces plottable
        return ShipTimeDerivatives3(states.x_dot, states.y_dot, states.psi_dot_deg, float(acc_in_world_frame[0]), float(acc_in_world_frame[1]), float(acc_in_world_frame[2])), GeneralizedForces(f_in_world[0], f_in_world[1], 0., 0., 0., f_in_world[2])

    def __repr__(self) -> str:
        return "{} Object".format(type(self).__name__)

    @property
    def inv_mass_matrix(self) -> np.ndarray:
        return self._inv_mass_matrix

    @property
    def coriolis_matrix(self) -> Callable:
        return self.__get_coriolis_matrix
    
    @property
    def coriolis_added_mass_matrix(self) -> Callable:
        return self.__get_coriolis_added_matrix
    
    @property
    def linear_damping_matrix(self) -> np.ndarray:
        return self._linear_damping_matrix

    @property
    def nonlinear_damping_matrix(self) -> Callable:
        return self.__get_nonlinear_damping_matrix
    
    @property
    def rotation_matrix(self) -> Callable:
        return self.__get_rotation_matrix

    @property
    def i_z(self) -> float:
        return self._i_z
    
    @property
    def mass(self) -> float:
        return self._mass
    
    @property
    def x_du(self) -> float:
        return self._x_du
    
    @property
    def y_dv(self) -> float:
        return self._y_dv
    
    @property
    def n_dr(self) -> float:
        return self._n_dr
    
    @property
    def t_surge(self) -> float:
        return self._t_surge
    
    @property
    def t_sway(self) -> float:
        return self._t_sway
    
    @property
    def t_yaw(self) -> float:
        return self._t_yaw
    
    @property
    def ku(self) -> float:
        return self._ku
    
    @property
    def kv(self) -> float:
        return self._kv
    
    @property
    def kr(self) -> float:
        return self._kr
    
    @property
    def params(self) -> ShipPhysicalParams:
        return self._params
    
    @property
    def x_g(self) -> float:
        return self._x_g
    
    @property
    def length(self) -> float:
        return self._length
    
    @property
    def width(self) -> float:
        return self._width
    
    @property
    def proj_area_front(self) -> float:
        return self._proj_area_front
    
    @property
    def proj_area_side(self) -> float:
        return self._proj_area_side
    
    @property
    def rho_a(self) -> float:
        return self._rho_a
    
# def test():
#     # Load the ship parameters from json file

#     ship_physics = ShipPhysics()
#     print(ship_physics)
#     print(ship_physics.inv_mass_matrix)
#     print(ship_physics.coriolis_matrix(1, 2, 3))
#     print(ship_physics.coriolis_added_mass_matrix(1, 2))
#     print(ship_physics.linear_damping_matrix)
#     print(ship_physics.nonlinear_damping_matrix(1, 2, 3))

#     state = ShipStates3(1., 2., 3., 4., 5., 6.)
#     wind = WindVector((0., 0.), vector=(1., 0.))

#     print(ship_physics.get_time_derivatives_and_forces(state, wind))

# if __name__ == "__main__":
#     test()