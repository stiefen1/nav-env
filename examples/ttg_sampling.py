def test() -> None:
    import os, matplotlib.pyplot as plt, numpy as np
    from nav_env.map.map import Map
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.wind.wind_source import UniformWindSource
    from nav_env.wind.stochastic import StochasticUniformWindSourceFactory
    from nav_env.ships.ship import Ship
    from nav_env.ships.states import States3
    from nav_env.simulation.integration import Euler
    from nav_env.risk.ttg import TTG
    from math import pi
    from scipy.spatial import Delaunay  
    from scipy.stats import gaussian_kde


    dt = 1.
    center = (43150, 6958000)
    size = (1500, 850)
    xlim = center[0]-size[0]/2, center[0]+size[0]/2
    ylim = center[1]-size[1]/2, center[1]+size[1]/2

    # Load map
    config_path = os.path.join('examples', 'config', 'config_seacharts.yaml')
    shore = Map(config_path, depth=0)
    obs = shore.get_obstacle_collection_in_window_from_enc(center=center, size=size)

    ship = Ship(States3(43093, 6957670, 20., -3., 10., 0.), integrator=Euler(dt), name="Ship1")
    # ship = Ship(States3(43040, 6957925, -50., -3., 10., 0.), integrator=Euler(dt), name="Ship1")


    # Wind
    uniform_wind = UniformWindSource(10, -10)

    # Environment
    env = Env(
        own_ships=[ship],
        wind_source=uniform_wind,
        shore=obs.as_list()
        )

    # Nominal TTG
    t_max = 200
    ttg = TTG(env)
    ttg_nominal = ttg.calculate(ship, t_max=t_max)
    
    env.plot(lim=((xlim[0], ylim[0]), (xlim[1], ylim[1])))
    plt.show()

    N = 1000
    wind_factory = StochasticUniformWindSourceFactory(10, -10, 4, pi)
    ttg_list = []
    w_list = []
    theta_list = []
    for n in range(N):
        uniform_wind = wind_factory()
        wind_vec = uniform_wind((0, 0))
        w, theta = wind_vec.intensity, wind_vec.direction
        env.wind_source = uniform_wind
        ttg = TTG(env)
        ttg_list.append(ttg.calculate(ship, t_max=t_max))
        w_list.append(w)
        theta_list.append(theta)
        print(f"ttg({n}) = {ttg_list[-1]:.2f}")

    # plt.hist(ttg_list)
    points = np.vstack((w_list, theta_list))
    mean = np.mean(points, axis=1)
    cov = np.cov(points)
    print("Mean: ", mean)
    print("Cov: \n", cov) # ATTENTION CA A L'AIR DE NE PAS CORRESPONDRE A CE QU'ON A MIS DANS StochasticUniformWindSourceFactory
    

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(w_list, theta_list, ttg_list, c=ttg_list)

    # Delaunay interpolation
    # w = np.array(w_list)
    # theta= np.array(theta_list)
    # points = np.vstack((w, theta)).T
    # tri = Delaunay(points)
    # ax.plot_trisurf(w_list, theta_list, ttg_list, triangles=tri.simplices, cmap='viridis', alpha=0.9)

    ax.set_xlabel('w')
    ax.set_ylabel('theta')
    ax.set_zlabel('TTG')
    plt.show()
    plt.close()

    kernel = gaussian_kde(ttg_list)
    kernel.integrate_box_1d(0, t_max)
    ttg_to_test_in_pdf = np.linspace(0, t_max, 100)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.hist(ttg_list, bins=200, range=(0, t_max))
    ax1.axvline(x=ttg_nominal, color='r', linestyle='--', label='$w=\\bar{w}$')
    ax1.axvline(x=np.mean(ttg_list), color='orange', linestyle='--', label='$\\bar{TTG}$')

    ax1.set_ylabel("Number of samples [-]")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(ttg_to_test_in_pdf, kernel.pdf(ttg_to_test_in_pdf))
    ax2.axvline(x=ttg_nominal, color='r', linestyle='--', label='$w=\\bar{w}$')
    ax2.axvline(x=np.mean(ttg_list), color='orange', linestyle='--', label='$\\bar{TTG}$')
    ax2.set_ylabel("PDF from Gaussian KDE [-]")
    plt.show()
    plt.close()
    

if __name__=="__main__":
    test()