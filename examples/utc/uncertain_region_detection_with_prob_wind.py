import os
from nav_env.map.map import Map
from nav_env.environment.environment import NavigationEnvironment as Env
from nav_env.wind.wind_source import UniformWindSource
from nav_env.wind.stochastic import StochasticUniformWindSourceFactory
from nav_env.ships.ship import Ship
from nav_env.ships.sailing_ship import SailingShip
from nav_env.control.path import TimeStampedWaypoints
from nav_env.ships.states import States3
from nav_env.simulation.integration import Euler
from nav_env.viz.matplotlib_screen import MatplotlibScreen as Screen
from nav_env.risk.ttg import TTG, TTGMaxWorsening, TTGWorstCase, TTGExpectation, TTGCurvature, TTGExpectedWorsening
from nav_env.risk.monitor import RiskMonitor
from nav_env.simulation.simulator import Simulator, SimulationRecord
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import multivariate_normal
import numpy as np

def plot_nominal() -> None:
    # # Save simulation record
    path_to_data_folder = os.path.join('recordings', 'uncertain_region_detection')
    filename = 'more_og_romsdal_nominal'
    path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'


    # Load map
    config_path = os.path.join('examples', 'config', 'config_seacharts.yaml')
    shore = Map(config_path, depth=0)
    
    # Limits
    center = (43150, 6958000)
    size = (1500, 850)
    xlim = center[0]-size[0]/2, center[0]+size[0]/2
    ylim = center[1]-size[1]/2, center[1]+size[1]/2

    # Route to follow
    v_mean = 10
    wpts = [
        States3(43900, 6957472),
        States3(43642, 6957472),
        States3(43080, 6957615),
        States3(43025, 6957906),
        States3(43234, 6958050),
        States3(43580, 6958265),
        States3(43425, 6958354)
        ]
    
    timestamped_wpts = []
    wpt_prev, t_prev = None, None
    for wpt in wpts:
        if wpt_prev is None:
            t_prev = 0
            wpt_prev = wpt
        else:
            dx = wpt.x - wpt_prev.x
            dy = wpt.y - wpt_prev.y
            dist = (dx**2 + dy**2)**0.5
            t_prev += dist / v_mean
            wpt_prev = wpt
        timestamped_wpts.append((t_prev, wpt_prev))
        print(t_prev)

# 0
# 25.8
# 83.79077512846331
# 113.40597392907918
# 138.78647825818973
# 179.5223346994136
# 197.39577868763254

# lim: 50-100 + 110-200 // 85-115
    # Generate stochastic wind source
    wind_source_factory = StochasticUniformWindSourceFactory(10, -20, 8, pi/6)

    # ship1 = Ship(States3(*center, -40., 10., 10., 0.), integrator=Euler(dt), name="Ship1")
    ship = SailingShip(length=40, width=10, pose_fn=TimeStampedWaypoints(timestamped_wpts))

    # Wind
    nominal_wind = wind_source_factory(nominal=True)

    # Environment
    nominal_env = Env(
        own_ships=[ship],
        wind_source=nominal_wind,
        shore=shore.as_list()
        )

    # Load it back
    nominal_record = SimulationRecord(path_to_existing_data=path_to_filename)

    # Plot environment
    ax = nominal_env.plot(lim=((xlim[0], ylim[0]), (xlim[1], ylim[1])), own_ships_physics={})

    sc = ax.scatter(nominal_record['SailingShip']['states']['x'], nominal_record['SailingShip']['states']['y'], c=nominal_record['SailingShip']['risks']['TTG'])
    plt.colorbar(sc)
    ax.axis('equal')
    plt.show()

def generate_data() -> None:
    N = 100 # Number of different wind conditions to test

    dt = 1.
    tf = 200
    center = (43150, 6958000)
    size = (1500, 850)
    xlim = center[0]-size[0]/2, center[0]+size[0]/2
    ylim = center[1]-size[1]/2, center[1]+size[1]/2

    # Load map
    config_path = os.path.join('examples', 'config', 'config_seacharts.yaml')
    shore = Map(config_path, depth=0)

    # Make data dir
    path_to_data_folder = os.path.join('recordings', 'uncertain_region_detection_2')
    os.makedirs(path_to_data_folder, exist_ok=True)


    # Route to follow
    v_mean = 10
    wpts = [
        States3(43900, 6957472),
        States3(43642, 6957472),
        States3(43080, 6957615),
        States3(43025, 6957906),
        States3(43234, 6958050),
        States3(43580, 6958265),
        States3(43425, 6958354)
        ]
    
    timestamped_wpts = []
    wpt_prev, t_prev = None, None
    for wpt in wpts:
        if wpt_prev is None:
            t_prev = 0
            wpt_prev = wpt
        else:
            dx = wpt.x - wpt_prev.x
            dy = wpt.y - wpt_prev.y
            dist = (dx**2 + dy**2)**0.5
            t_prev += dist / v_mean
            wpt_prev = wpt
        timestamped_wpts.append((t_prev, wpt_prev))

    sigma_x, sigma_y = 8, pi/6

    # Generate stochastic wind source
    wind_source_factory = StochasticUniformWindSourceFactory(10, -20, sigma_x, sigma_y)
    nominal_vec = wind_source_factory(nominal=True)((0, 0))
    
    mu_x, mu_y = nominal_vec.intensity, nominal_vec.direction
    print(nominal_vec.vx, nominal_vec.vy, mu_x, mu_y)
    rv = multivariate_normal([mu_x, mu_y], [[sigma_x**2, 0], [0, sigma_y**2]])

    # ship1 = Ship(States3(*center, -40., 10., 10., 0.), integrator=Euler(dt), name="Ship1")
    ship = SailingShip(length=40, width=10, pose_fn=TimeStampedWaypoints(timestamped_wpts))

    # Wind
    # uniform_wind = UniformWindSource(10, -20)
    nominal_wind = wind_source_factory(nominal=True)

    # Environment
    nominal_env = Env(
        own_ships=[ship],
        wind_source=nominal_wind,
        shore=shore.as_list()
        )

    nominal_sim = Simulator(env=nominal_env, monitor=RiskMonitor([TTG]))
    nominal_sim.run(tf=tf, record_results_dt=1, dt=dt)

    # # Save simulation record
    filename = 'more_og_romsdal_nominal'
    path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'
    nominal_sim.record.save(path_to_filename)

    pdf_list_of_values = []
    for i in range(N):
        ship = SailingShip(length=40, width=10, pose_fn=TimeStampedWaypoints(timestamped_wpts))

        wind = wind_source_factory()
        pdf_list_of_values.append(rv.pdf((wind((0, 0)).intensity, wind((0, 0)).direction)))
        print("Wind: ", wind((0, 0)).intensity, wind((0, 0)).direction, pdf_list_of_values[-1])
        # Environment
        env_i = Env(
            own_ships=[ship],
            wind_source=wind,
            shore=shore.as_list()
            )

        sim_i = Simulator(env=env_i, monitor=RiskMonitor([TTG]))
        sim_i.run(tf=tf, record_results_dt=1, dt=dt)

        # # Save simulation record
        filename = 'more_og_romsdal_' + str(i+1)
        path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'
        sim_i.record.save(path_to_filename)

    np.savez(os.path.join(path_to_data_folder, 'pdf'), pdf=pdf_list_of_values)

    # Load it back
    # new_sim_record = SimulationRecord(path_to_existing_data=path_to_filename)

    # # Plot environment
    # ax = env.plot(lim=((xlim[0], ylim[0]), (xlim[1], ylim[1])), own_ships_physics={})

    # sc = ax.scatter(new_sim_record['SailingShip']['states']['x'], new_sim_record['SailingShip']['states']['y'], c=new_sim_record['SailingShip']['risks']['TTG'])
    # plt.colorbar(sc)
    # ax.axis('equal')
    # plt.show()

def schmitt_trigger(x, low:float=0.05, high:float=0.25) -> list[tuple]:
    lim = []
    detection = False
    for idx, j in enumerate(x):
        if not detection and j>=high:
            start_idx = idx
            detection=True
        elif detection and j<=low:
            stop_idx = idx
            lim.append((start_idx, stop_idx))
            detection=False
    if detection:
        stop_idx = idx
        lim.append((start_idx, stop_idx))
        detection = False
    return lim

def group_close_limit(lim:list[tuple], min_dist:float=25, depth:int=0, max_recursion_depth:int=100) -> list[tuple]:
    prev_lim = None
    assert depth < max_recursion_depth, f"Max recursion depth exceeded in group_close_limit method"

    for i, lim_i in enumerate(lim):
        if prev_lim is None:
            prev_lim = lim_i 
            continue
        elif lim_i[0] - prev_lim[1] <= min_dist:
            # Means we have to remove lim_i and prev_lim from lim
            # And replace it by (prev_lim[0], lim_i[1])
            lim.pop(i)
            lim.pop(i-1)
            lim.insert(i-1, (prev_lim[0], lim_i[1]))
            group_close_limit(lim, min_dist=min_dist, depth=depth+1)
            break
        prev_lim = lim_i

    return lim

def remove_small_lim(lim:list[tuple], min_width:int=25) -> list[tuple]:
    new_lim = []
    for lim_i in lim:
        if lim_i[1] - lim_i[0] >= min_width:
            new_lim.append(lim_i)

    return new_lim

def plot_data():
    import numpy as np
    import sys, pathlib
    sys.path.append(os.path.join(pathlib.Path(__file__).parent.parent.parent, 'submodules', 'UTCRisk'))
    from RiskModelJSON import RiskModel

    config_path = os.path.join(os.path.curdir, 'submodules', 'UTCRisk', 'ship_config.json')
    risk_model = RiskModel(config_path)

    N = 100
    path_to_data_folder = os.path.join('recordings', 'uncertain_region_detection_2')

    # Nominal recording
    filename = 'more_og_romsdal_nominal'
    path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'
    record_nominal = SimulationRecord(path_to_existing_data=path_to_filename)
    ttg_nominal = np.array(record_nominal['SailingShip']['risks']['TTG'])
    r_nominal = []
    for ttg in ttg_nominal:
        # mode = risk_model.select_mso_mode(ttg)
        r_nominal.append(risk_model.compute_total_risk(ttg, "MEC"))

    r_nominal = np.array(r_nominal).squeeze()

    # Array to receive \Delta TTG_i, i=1, ..., N
    delta_r_array = np.zeros(shape=(N, ttg_nominal.shape[0]))
    r_array = np.zeros_like(delta_r_array)
    pdf_values = np.load(os.path.join(path_to_data_folder, 'pdf.npz'))['pdf']
    delta_r_array_pdf_scaled = np.zeros_like(delta_r_array)

    # i-th recording
    for i in range(N):
        # Load TTG data
        filename = 'more_og_romsdal_' + str(i+1)
        path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'
        record_i = SimulationRecord(path_to_existing_data=path_to_filename)
        ttg_i = np.array(record_i['SailingShip']['risks']['TTG'])

        # Computing risk corresponding to each ttg value
        r_i = []
        r_i_pdf_scaled = []
        for ttg in ttg_i:
            r = risk_model.compute_total_risk(ttg, "MEC")
            r_i.append(r)
        r_i = np.array(r_i).squeeze()

        # Compute difference between r_i and nominal. if delta > 0 risk is higher
        delta_r_i = r_i - r_nominal
        r_array[i] = r_i.squeeze()
        delta_r_array[i] = delta_r_i.squeeze()
        delta_r_array_pdf_scaled[i] = pdf_values[i] * delta_r_i.squeeze() / np.sum(pdf_values)

    
    print(pdf_values.shape)

    signal = delta_r_array.std(axis=0)
    trigger = remove_small_lim(group_close_limit(schmitt_trigger(signal, low=0.1, high=0.15), min_dist=5), min_width=25)
    # trigger = group_close_limit(schmitt_trigger(signal, low=0.1, high=0.15), min_dist=5)
    # trigger = schmitt_trigger(signal, low=0.1, high=0.15)

    signal2 =  delta_r_array_pdf_scaled.max(axis=0)
    # trigger2 = remove_small_lim(group_close_limit(schmitt_trigger(signal2, low=0.05, high=0.05), min_dist=5), min_width=25)
    low, high = 0.001, 0.003
    trigger2 = remove_small_lim(group_close_limit(schmitt_trigger(signal2, low=low, high=high), min_dist=15), min_width=5)

    plt.figure()
    plt.plot(signal2, color='black', label="$\\mathbb{E}\{ \\Delta R_G \}$")
    # plt.plot(signal2 + std2, color='red')
    # plt.plot(signal2 - std2, color='red')
    # plt.plot(signal2 + 2*std2, color='red')
    # plt.plot(signal2 - 2*std2, color='red')
    # plt.plot(signal2 + 3*std2, color='red')
    # plt.plot(signal2 - 3*std2, color='red')
    for i, lim_i in enumerate(trigger2):
        plt.axvline(x=lim_i[0], color='black', linestyle='--', linewidth=1)
        plt.axvline(x=lim_i[1], color='black', linestyle='--', linewidth=1)
        plt.axhline(y=low, color='blue', linestyle='--', linewidth=1, label="$s_0$ (Schmitt Trigger)" if i==0 else None)
        plt.axhline(y=high, color='red', linestyle='--', linewidth=1, label="$s_f$ (Schmitt Trigger)" if i==0 else None)
        plt.fill_betweenx(np.linspace(0,0.03, 10), 10*[lim_i[0]], 10*[lim_i[1]], color='black', alpha=0.08)
        plt.text(100, -0.5, "Uncertain\nRegion", horizontalalignment='center', fontsize=8)
    plt.xlabel("Time [s]")
    plt.ylabel("$\\Delta R_G$ [-]")
    plt.legend()
    # plt.ylim([-0.9, 0.4])
    plt.xlim([0, 200])
    plt.show()
    plt.close()

    print(trigger)

    plt.figure()
    plt.plot(signal, color='black')
    plt.xlabel("t [s]")
    plt.ylabel("$\\sigma_{TTG}$ [s]")

    for i, lim_i in enumerate(trigger):
        plt.axvline(x=lim_i[0], color='black', linestyle='--', linewidth=1)
        plt.axvline(x=lim_i[1], color='black', linestyle='--', linewidth=1)
        plt.fill_betweenx(np.linspace(0, 1, 10), 10*[lim_i[0]], 10*[lim_i[1]], color='black', alpha=0.08)
        #plt.text(np.mean(lim_i), 0.05, f"Uncertain\nRegion {i}", fontdict={'color':'black'}, horizontalalignment='center')
        
    plt.ylim([0., 0.8])
    # plt.plot(delta_r_array.mean(axis=0), color='b')
    # plt.fill_between(np.linspace(0, delta_r_array.shape[1], delta_r_array.shape[1]), delta_r_array.mean(axis=0)-delta_r_array.std(axis=0), delta_r_array.mean(axis=0)+delta_r_array.std(axis=0), alpha=0.5, color='b')
    plt.show()
    plt.close()

    # # Schmitt trigger
    # schmitt_high = 0.25
    # schmitt_low = 0.05   
    # for i in range(N):
    #     delta_r_i = delta_r_array[i].squeeze()
        
    #     plt.figure()
    #     plt.plot(delta_r_array[i].T)
        
    #     for lim_i in remove_small_lim(group_close_limit(schmitt_trigger(delta_r_i, low=schmitt_low, high=schmitt_high), min_dist=25), min_width=25):
    #         plt.axvline(x=lim_i[0], color='g')
    #         plt.axvline(x=lim_i[1], color='r')
    #         print(lim_i)

    #     # plt.plot(delta_r_array.max(axis=0) + delta_r_array.min(axis=0))
    #     # plt.plot(delta_r_array.mean(axis=0) + 3*delta_r_array.std(axis=0))
    #     # plt.plot(r_nominal, '--r')
    #     plt.show()
    #     plt.close()

def test():
    import numpy as np
    _, ax = plt.subplots(subplot_kw={'projection': '3d'})

    #Parameters to set
    mu_x = 22.36
    sigma_x = 8

    mu_y = 3.6
    sigma_y = pi/6

    #Create grid and multivariate normal
    N = 3
    x = np.linspace(mu_x-N*sigma_x,mu_x+N*sigma_x,100)
    y = np.linspace(mu_y-N*sigma_y,mu_y+N*sigma_y,100)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal([mu_x, mu_y], [[sigma_x**2, 0], [0, sigma_y**2]])

    points_to_check = np.array([
        [10, 20],
        [5, 20],
        [10, 22],
        [1, 20],
        [-5, 21]
    ])

    # ax.scatter(points_to_check[:,0], points_to_check[:, 1], rv.pdf(points_to_check))

    #Make a 3D plot
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    ax.set_xlabel('Wind Intensity [m/s]')
    ax.set_ylabel('Wind Direction [rad]')
    ax.set_zlabel('pdf [-]')

    plt.show()

if __name__ == "__main__":
    # generate_data()
    # plot_data()
    plot_nominal()
    # test()