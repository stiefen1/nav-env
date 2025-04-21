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

def plot_nominal() -> None:
    # # Save simulation record
    path_to_data_folder = os.path.join('recordings', 'uncertain_region_detection')
    filename = 'more_og_romsdal_nominal'
    path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'

    import sys, pathlib
    sys.path.append(os.path.join(pathlib.Path(__file__).parent.parent.parent, 'submodules', 'UTCRisk'))
    # from RiskModelJSON import RiskModel

    config_path = os.path.join(os.path.curdir, 'submodules', 'UTCRisk', 'ship_config.json')
    risk_model = RiskModel(config_path)


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
    ax = nominal_env.plot(lim=((xlim[0], ylim[0]), (xlim[1], ylim[1])), own_ships_physics={'enveloppe':1})

    risk = []
    for ttg in nominal_record['SailingShip']['risks']['TTG']:
        risk.append(risk_model.compute_total_risk(ttg, "MEC"))
    
    sc = ax.scatter(nominal_record['SailingShip']['states']['x'], nominal_record['SailingShip']['states']['y'], c=risk, s=5)
    plt.colorbar(sc, label="$R_G$ [-]")

    dx = [0,  10, -100, -100, -60, 45, 0]
    dy = [50, 50,  0,   0,  50, 0, 50]
    for i, wpt in enumerate(wpts):
        t = ax.text(wpt.x+dx[i], wpt.y+dy[i], f"$w_{i}$")
        ax.scatter(*wpt.xy, color='black')
        t.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=1))

    # ax.axis('equal')
    ax.locator_params(axis='both', nbins=7)
    ax.set_xlim([42700, 44000])
    ax.set_ylim([6957400, 6958500])
    ax.legend()
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
    path_to_data_folder = os.path.join('recordings', 'uncertain_region_detection')
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

    # Generate stochastic wind source
    wind_source_factory = StochasticUniformWindSourceFactory(10, -20, 8, pi/6)

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

    for i in range(N):
        ship = SailingShip(length=40, width=10, pose_fn=TimeStampedWaypoints(timestamped_wpts))

        # Environment
        env_i = Env(
            own_ships=[ship],
            wind_source=wind_source_factory(),
            shore=shore.as_list()
            )

        sim_i = Simulator(env=env_i, monitor=RiskMonitor([TTG]))
        sim_i.run(tf=tf, record_results_dt=1, dt=dt)

        # # Save simulation record
        filename = 'more_og_romsdal_' + str(i+1)
        path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'
        sim_i.record.save(path_to_filename)

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
    path_to_data_folder = os.path.join('recordings', 'uncertain_region_detection')

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

    # i-th recording
    for i in range(N):
        # Load TTG data
        filename = 'more_og_romsdal_' + str(i+1)
        path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'
        record_i = SimulationRecord(path_to_existing_data=path_to_filename)
        ttg_i = np.array(record_i['SailingShip']['risks']['TTG'])

        # Computing risk corresponding to each ttg value
        r_i = []
        for ttg in ttg_i:
            r_i.append(risk_model.compute_total_risk(ttg, "MEC"))
        r_i = np.array(r_i).squeeze()

        # Compute difference between r_i and nominal. if delta > 0 risk is higher
        delta_r_i = r_i - r_nominal
        r_array[i] = r_i.squeeze()
        delta_r_array[i] = delta_r_i.squeeze()

    signal = delta_r_array.std(axis=0)
    trigger = remove_small_lim(group_close_limit(schmitt_trigger(signal, low=0.1, high=0.15), min_dist=5), min_width=25)
    # trigger = group_close_limit(schmitt_trigger(signal, low=0.1, high=0.15), min_dist=5)
    # trigger = schmitt_trigger(signal, low=0.1, high=0.15)

    signal2 =  delta_r_array.mean(axis=0) + 1*delta_r_array.std(axis=0)
    # trigger2 = remove_small_lim(group_close_limit(schmitt_trigger(signal2, low=0.05, high=0.05), min_dist=5), min_width=25)
    low, high = 0., 0.3
    trigger2 = remove_small_lim(group_close_limit(schmitt_trigger(signal2, low=low, high=high), min_dist=20), min_width=10)

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
        plt.fill_betweenx(np.linspace(-1, 1, 10), 10*[lim_i[0]], 10*[lim_i[1]], color='black', alpha=0.08)
        plt.text(100, -0.5, "Uncertain\nRegion", horizontalalignment='center', fontsize=8)

    plt.text(98, -0.1, "Uncertain\nRegion", horizontalalignment='center', fontsize=8)
    plt.text(165, -0.1, "Uncertain\nRegion", horizontalalignment='center', fontsize=8)

    plt.xlabel("Time [s]")
    plt.ylabel("$\\mathbb{E}\{\\Delta R_G\} + 1\\cdot\\sigma_R$ [-]")
    plt.legend()
    plt.ylim([-0.2, 0.8])
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

def plot_pdns():
    import numpy as np
    import sys, pathlib
    sys.path.append(os.path.join(pathlib.Path(__file__).parent.parent.parent, 'submodules', 'UTCRisk'))
    from RiskModelJSON import RiskModel

    config_path = os.path.join(os.path.curdir, 'submodules', 'UTCRisk', 'ship_config.json')
    risk_model = RiskModel(config_path)

    N = 100
    path_to_data_folder = os.path.join('recordings', 'uncertain_region_detection')

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
    sigma_degradation = np.zeros_like(delta_r_array)

    # i-th recording
    for i in range(N):
        # Load TTG data
        filename = 'more_og_romsdal_' + str(i+1)
        path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'
        record_i = SimulationRecord(path_to_existing_data=path_to_filename)
        ttg_i = np.array(record_i['SailingShip']['risks']['TTG'])

        # Computing risk corresponding to each ttg value
        r_i = []
        for ttg in ttg_i:
            r_i.append(risk_model.compute_total_risk(ttg, "MEC"))
        r_i = np.array(r_i).squeeze()


        # Compute difference between r_i and nominal. if delta > 0 risk is higher
        delta_r_i = r_i - r_nominal
        r_array[i] = r_i.squeeze()
        delta_r_array[i] = delta_r_i.squeeze()

    val_at_std = r_array.mean(axis=0) + r_array.std(axis=0)
    val_at_std = np.minimum(np.ones_like(val_at_std)*1.7, val_at_std)                                                                                                   ###### ATTENTION, IL FAUDRAIT LIMITER LE MEAN(RISK) + STD(RISK) POUR NE PAS QU'IL PUISSE DEPASSER LA VALEUR MAX DU RISQUE
    delta_r_array = val_at_std - r_nominal


    signal2 =  delta_r_array#delta_r_array.mean(axis=0) + 1*delta_r_array.std(axis=0)
    # trigger2 = remove_small_lim(group_close_limit(schmitt_trigger(signal2, low=0.05, high=0.05), min_dist=5), min_width=25)
    low, high = 0., 0.3
    trigger2 = remove_small_lim(group_close_limit(schmitt_trigger(signal2, low=low, high=high), min_dist=20), min_width=10)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(signal2, color='black', label="$\\mathbb{E}\{\\Delta R_G\} + 1\\cdot\\sigma_R$ [-]")
    for i, lim_i in enumerate(trigger2):
        axs[0].axvline(x=lim_i[0], color='green', linestyle='--', linewidth=1)
        axs[0].axvline(x=lim_i[1], color='green', linestyle='--', linewidth=1)
        axs[0].axhline(y=low, color='blue', linestyle='--', linewidth=1, label="$s_0$ (Schmitt Trigger)" if i==0 else None)
        axs[0].axhline(y=high, color='red', linestyle='--', linewidth=1, label="$s_f$ (Schmitt Trigger)" if i==0 else None)
        axs[0].fill_betweenx(np.linspace(-1, 1, 10), 10*[lim_i[0]], 10*[lim_i[1]], color='green', alpha=0.08)
        axs[0].text(100, -0.5, "Uncertain\nRegion", horizontalalignment='center', fontsize=8)

    axs[0].text(98, -0.1, "Uncertain\nRegion", horizontalalignment='center', fontsize=8)
    axs[0].text(165, -0.1, "Uncertain\nRegion", horizontalalignment='center', fontsize=8)

    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("$\\Delta R_G$ [-]")
    axs[0].legend()
    axs[0].set_ylim([-0.2, 0.8])
    axs[0].set_xlim([0, 200])
    # plt.show()
    # plt.close()

    ############################################# PLOT MAP WITH UNCERTAIN REGIONS #############################################

    # # Save simulation record
    path_to_data_folder = os.path.join('recordings', 'uncertain_region_detection')
    filename = 'more_og_romsdal_nominal'
    path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'

    config_path = os.path.join(os.path.curdir, 'submodules', 'UTCRisk', 'ship_config.json')
    risk_model = RiskModel(config_path)


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
    ax = nominal_env.plot(lim=((xlim[0], ylim[0]), (xlim[1], ylim[1])), own_ships_physics={'enveloppe':1}, ax=axs[1])
    
    x, y = nominal_record['SailingShip']['states']['x'], nominal_record['SailingShip']['states']['y']
    t0 = 0

    dx = [0,  10, -100, -100, -60, 45, 0]
    dy = [50, 50,  0,   0,  50, 0, 60]
    for i, wpt in enumerate(wpts):
        t = ax.text(wpt.x+dx[i], wpt.y+dy[i], f"$w_{i}$")
        ax.scatter(*wpt.xy, color='black')
        t.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=1))

    ax.plot(x, y, '--', color='black', label="Initial route")


    dx = [(40, 40),    (40, -100)]
    dy = [(40, -40), (-60, -10)]
    for i, lim_i in enumerate(trigger2):

        list_of_w_in_uncertain_region_i = [ship.pose_fn(lim_i[0]).xy]
        for t, wpt in timestamped_wpts:
            if t >= lim_i[0] and t <= lim_i[1]:
                list_of_w_in_uncertain_region_i.append(ship.pose_fn(t).xy)
        list_of_w_in_uncertain_region_i.append(ship.pose_fn(lim_i[1]).xy)
        ax.plot(*zip(*list_of_w_in_uncertain_region_i), color='green', linewidth=5, alpha=0.5, label=f"Uncertain region" if i==0 else None)

        ax.scatter(x[lim_i[0]], y[lim_i[0]], marker='x', s=50, color='green', linewidth=2)
        t1 = ax.text(x[lim_i[0]] + dx[i][0], y[lim_i[0]] + dy[i][0], f"$w^{i+1}_0$", color='green')
        t1.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=1))
        ax.scatter(x[lim_i[1]], y[lim_i[1]], marker='x', s=50, color='green', linewidth=3)
        t2 = ax.text(x[lim_i[1]] + dx[i][1], y[lim_i[1]] + dy[i][1], f"$w^{i+1}_f$", color='green')
        t2.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=1))
        tf = lim_i[0]
        ax.plot(x[t0:tf], y[t0:tf], '--', color='black')
        t0 = lim_i[1]

    # ax.plot(x[t0:-1], y[t0:-1], '--', color='black', label="Initial route")

    # ax.axis('equal')
    ax.locator_params(axis='both', nbins=7)
    ax.set_xlim([42700, 44000])
    ax.set_ylim([6957400, 6958500])
    ax.legend()
    plt.show()

def plot_ed():
    import numpy as np
    import sys, pathlib
    sys.path.append(os.path.join(pathlib.Path(__file__).parent.parent.parent, 'submodules', 'UTCRisk'))
    from RiskModelJSON import RiskModel

    config_path = os.path.join(os.path.curdir, 'submodules', 'UTCRisk', 'ship_config.json')
    risk_model = RiskModel(config_path)

    N = 100
    path_to_data_folder = os.path.join('recordings', 'uncertain_region_detection')

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

    # i-th recording
    for i in range(N):
        # Load TTG data
        filename = 'more_og_romsdal_' + str(i+1)
        path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'
        record_i = SimulationRecord(path_to_existing_data=path_to_filename)
        ttg_i = np.array(record_i['SailingShip']['risks']['TTG'])

        # Computing risk corresponding to each ttg value
        r_i = []
        for ttg in ttg_i:
            r_i.append(risk_model.compute_total_risk(ttg, "MEC"))
        r_i = np.array(r_i).squeeze()

        # Compute difference between r_i and nominal. if delta > 0 risk is higher
        delta_r_i = r_i - r_nominal
        r_array[i] = r_i.squeeze()
        delta_r_array[i] = delta_r_i.squeeze()


    signal2 =  delta_r_array.mean(axis=0)
    # trigger2 = remove_small_lim(group_close_limit(schmitt_trigger(signal2, low=0.05, high=0.05), min_dist=5), min_width=25)
    low, high = -0.1, 0.1
    trigger2 = remove_small_lim(group_close_limit(schmitt_trigger(signal2, low=low, high=high), min_dist=20), min_width=10)

    # plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(signal2, color='black', label="$\\mathbb{E}\{\\Delta R_G\}$")
    for i, lim_i in enumerate(trigger2):
        axs[0].axvline(x=lim_i[0], color='green', linestyle='--', linewidth=1)
        axs[0].axvline(x=lim_i[1], color='green', linestyle='--', linewidth=1)
        axs[0].axhline(y=low, color='blue', linestyle='--', linewidth=1, label="$s_0$ (Schmitt Trigger)" if i==0 else None)
        axs[0].axhline(y=high, color='red', linestyle='--', linewidth=1, label="$s_f$ (Schmitt Trigger)" if i==0 else None)
        axs[0].fill_betweenx(np.linspace(-1, 1, 10), 10*[lim_i[0]], 10*[lim_i[1]], color='green', alpha=0.08)
        axs[0].text(100, -0.5, "Uncertain\nRegion", horizontalalignment='center', fontsize=8)

    # plt.text(98, -0.1, "Uncertain\nRegion", horizontalalignment='center', fontsize=8)
    # plt.text(165, -0.1, "Uncertain\nRegion", horizontalalignment='center', fontsize=8)

    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("$\\Delta R_G$ [-]")
    axs[0].legend()
    axs[0].set_ylim([-0.9, 0.4])
    axs[0].set_xlim([0, 200])
    # plt.show()
    # plt.close()

    ############################################# PLOT MAP WITH UNCERTAIN REGIONS #############################################

    # # Save simulation record
    path_to_data_folder = os.path.join('recordings', 'uncertain_region_detection')
    filename = 'more_og_romsdal_nominal'
    path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'

    config_path = os.path.join(os.path.curdir, 'submodules', 'UTCRisk', 'ship_config.json')
    risk_model = RiskModel(config_path)


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
    ax = nominal_env.plot(lim=((xlim[0], ylim[0]), (xlim[1], ylim[1])), own_ships_physics={'enveloppe':1}, ax=axs[1])

    x, y = nominal_record['SailingShip']['states']['x'], nominal_record['SailingShip']['states']['y']
    t0 = 0

    dx = [0,  10, -100, -100, -60, 45, 0]
    dy = [50, 50,  0,   0,  50, 0, 60]
    for i, wpt in enumerate(wpts):
        t = ax.text(wpt.x+dx[i], wpt.y+dy[i], f"$w_{i}$")
        ax.scatter(*wpt.xy, color='black')
        t.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=1))

    dx = [(40, 40),    (40, 40)]
    dy = [(40, -40), (-60, -50)]
    for i, lim_i in enumerate(trigger2):

        list_of_w_in_uncertain_region_i = [ship.pose_fn(lim_i[0]).xy]
        for t, wpt in timestamped_wpts:
            if t >= lim_i[0] and t <= lim_i[1]:
                list_of_w_in_uncertain_region_i.append(ship.pose_fn(t).xy)
        list_of_w_in_uncertain_region_i.append(ship.pose_fn(lim_i[1]).xy)
        ax.plot(*zip(*list_of_w_in_uncertain_region_i), color='green', linewidth=5, alpha=0.5, label=f"Uncertain region")

        ax.scatter(x[lim_i[0]], y[lim_i[0]], marker='x', s=50, color='green', linewidth=2)
        t1 = ax.text(x[lim_i[0]] + dx[i][0], y[lim_i[0]] + dy[i][0], f"$w^{i+1}_0$", color='green')
        t1.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=1))
        ax.scatter(x[lim_i[1]], y[lim_i[1]], marker='x', s=50, color='green', linewidth=3)
        t2 = ax.text(x[lim_i[1]] + dx[i][1], y[lim_i[1]] + dy[i][1], f"$w^{i+1}_f$", color='green')
        t2.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=1))
        tf = lim_i[0]
        ax.plot(x[t0:tf], y[t0:tf], '--', color='black')
        t0 = lim_i[1]

        

    ax.plot(x[t0:-1], y[t0:-1], '--', color='black', label="Initial route")

    # ax.axis('equal')
    ax.locator_params(axis='both', nbins=7)
    ax.set_xlim([42700, 44000])
    ax.set_ylim([6957400, 6958500])
    ax.legend()
    plt.show()

def plot_all_risk_along_traj():
    import numpy as np
    import sys, pathlib
    sys.path.append(os.path.join(pathlib.Path(__file__).parent.parent.parent, 'submodules', 'UTCRisk'))
    from RiskModelJSON import RiskModel

    config_path = os.path.join(os.path.curdir, 'submodules', 'UTCRisk', 'ship_config.json')
    risk_model = RiskModel(config_path)

    N = 100
    path_to_data_folder = os.path.join('recordings', 'uncertain_region_detection')

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

    v = UniformWindSource(10, -20)((0, 0))
    print(f"I: {v.intensity}, D: {v.direction}")
    10, -20, 8, pi/6    

    # i-th recording
    for i in range(N):
        # Load TTG data
        filename = 'more_og_romsdal_' + str(i+1)
        path_to_filename = os.path.join(path_to_data_folder, filename) + '.csv'
        record_i = SimulationRecord(path_to_existing_data=path_to_filename)
        ttg_i = np.array(record_i['SailingShip']['risks']['TTG'])

        # Computing risk corresponding to each ttg value
        r_i = []
        for ttg in ttg_i:
            r_i.append(risk_model.compute_total_risk(ttg, "MEC"))
        r_i = np.array(r_i).squeeze()

        # Compute difference between r_i and nominal. if delta > 0 risk is higher
        delta_r_i = r_i - r_nominal
        r_array[i] = r_i.squeeze()
        delta_r_array[i] = delta_r_i.squeeze()


    signal2 =  delta_r_array.mean(axis=0) + 1*delta_r_array.std(axis=0)
    low, high = 0., 0.3
    trigger2 = remove_small_lim(group_close_limit(schmitt_trigger(signal2, low=low, high=high), min_dist=20), min_width=10)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # All traj
    # ax.plot(r_array.T,  alpha=0.5)

    # Exp + 3*std
    ax.plot(r_array.mean(axis=0), color='black', label="$\\mathbb{E}\{\\mathbf{R_{\\Theta,E}(x, v)}\}$")
    ax.fill_between(np.linspace(0, 200, 201), r_array.mean(axis=0) - r_array.std(axis=0), r_array.mean(axis=0) + r_array.std(axis=0), alpha=0.08, color='black', label="$\\pm 1, 2, 3\sigma_R$")
    ax.fill_between(np.linspace(0, 200, 201), r_array.mean(axis=0) - 2*r_array.std(axis=0), r_array.mean(axis=0) + 2*r_array.std(axis=0), alpha=0.08, color='black')
    ax.fill_between(np.linspace(0, 200, 201), r_array.mean(axis=0) - 3*r_array.std(axis=0), r_array.mean(axis=0) + 3*r_array.std(axis=0), alpha=0.08, color='black')



    # Nom + Exp + 3*std
    # ax.plot(r_array.mean(axis=0), color='black', label="$\\mathbb{E}\{\\mathbf{R_{\\Theta,E}(x, v)}\}$")
    # ax.plot(r_nominal, color='red', linestyle='--', label="$\\mathbf{R_{\\Theta,E^{nom}}(x, v)}$")
    # ax.fill_between(np.linspace(0, 200, 201), r_array.mean(axis=0) - r_array.std(axis=0), r_array.mean(axis=0) + r_array.std(axis=0), alpha=0.08, color='black', label="$\\pm 1, 2, 3\sigma_R$")
    # ax.fill_between(np.linspace(0, 200, 201), r_array.mean(axis=0) - 2*r_array.std(axis=0), r_array.mean(axis=0) + 2*r_array.std(axis=0), alpha=0.08, color='black')
    # ax.fill_between(np.linspace(0, 200, 201), r_array.mean(axis=0) - 3*r_array.std(axis=0), r_array.mean(axis=0) + 3*r_array.std(axis=0), alpha=0.08, color='black')


    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$R_G$ [-]")
    ax.set_ylim([-1.5, 3.5])
    ax.legend()
    plt.show()
    plt.close()

if __name__ == "__main__":
    # generate_data()
    # plot_data()
    # plot_pdns()
    # plot_ed()
    plot_nominal()
    # plot_all_risk_along_traj()