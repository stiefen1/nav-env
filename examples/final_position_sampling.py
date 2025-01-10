import os
BASE_PATH = os.path.join('M:', 'Documents', 'Kongsberg', '01_Enhancing_Risk_Awareness_In_Autonomous_Ship_Navigation', 'figures', 'wind')


def generate_data() -> None:
    import matplotlib.pyplot as plt, numpy as np
    from nav_env.map.map import Map
    from nav_env.environment.environment import NavigationEnvironment as Env
    from nav_env.wind.wind_source import UniformWindSource
    from nav_env.wind.stochastic import StochasticUniformWindSourceFactory
    from nav_env.ships.ship import Ship
    from nav_env.ships.states import States3
    from nav_env.simulation.integration import Euler
    from nav_env.risk.ttg import TTG
    from math import pi
    from nav_env.wind.wind_vector import WindVector

    dt = 1.
    center = (43150, 6958000)
    size = (1500, 850)
    xlim = center[0]-size[0]/2, center[0]+size[0]/2
    ylim = center[1]-size[1]/2, center[1]+size[1]/2

    # Load map
    config_path = os.path.join('examples', 'config', 'config_seacharts.yaml')
    shore = Map(config_path, depth=0)
    obs = shore.get_obstacle_collection_in_window_from_enc(center=center, size=size)

    ship = Ship(States3(43150, 6957670, 0., 0., 10., 0.), integrator=Euler(dt), name="Ship1")

    for direction in np.linspace(np.pi, 2*np.pi, 19):

        direction_dir = os.path.join(BASE_PATH, f"{direction*180/np.pi:.0f}deg")
        data_dir = os.path.join(direction_dir, 'DATA')
        img_dir = os.path.join(direction_dir, 'IMG')

        try:
            os.mkdir(direction_dir)
            os.mkdir(data_dir)
            os.mkdir(img_dir)
        except FileExistsError:
            pass

        # Wind
        w = WindVector((0, 0), direction=direction, intensity=15)
        #wx, wy = 0, -15 # i.e. 15 m/s = 30 knots
        wx, wy = w.vx, w.vy
        uniform_wind = UniformWindSource(wx, wy)
    
        # Environment
        env = Env(
            own_ships=[ship],
            wind_source=uniform_wind
            )

        wind_factory = StochasticUniformWindSourceFactory(wx, wy, 4, pi/4)

        for t_max in np.linspace(10, 200, 20):
            filename = f"{t_max:.0f}s"
            print(f"Working on {direction*180/np.pi:.0f}deg , {t_max:.0f}s")
            
            ttg_list = []
            final_x_list = []
            final_y_list = []
            w_list = []
            theta_list = []
            N = 100
            for n in range(N):
                perturbed_uniform_wind = wind_factory()
                perturbed_wind_vec = perturbed_uniform_wind((0, 0))
                w, theta = perturbed_wind_vec.intensity, perturbed_wind_vec.direction
                env.wind_source = perturbed_uniform_wind
                ttg_metric = TTG(env)
                ttg, state = ttg_metric.calculate(ship, output_final_state=True, t_max=t_max)
                ttg_list.append(ttg)
                final_x_list.append(state.x)
                final_y_list.append(state.y)
                w_list.append(w)
                theta_list.append(theta)

            # Pourquoi pas observer l'évolution de la variance avec le temps en fonction de la vitesse initiale, angle
            # d'incidence du vent et essayer d'en déduire une formule ?


            fig = plt.figure()
            ax = fig.add_subplot()
            env.wind_source = uniform_wind
            env.plot(lim=((xlim[0], ylim[0]), (xlim[1], ylim[1])), ax=ax, own_ships_physics={'enveloppe':1, 'velocity':1})
            ax.scatter(final_x_list, final_y_list, c='red', alpha=0.8)
            ax.set_title(f"t = {t_max:.2f} s , wind = 30 knots , v0 = 20 knots")

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.axis('equal')
            plt.savefig(os.path.join(img_dir, filename))
            plt.close()

            # Save as numpy arrays
            final_x_np = np.array(final_x_list) 
            final_y_np = np.array(final_y_list) 
            w_np = np.array(w_list) 
            theta_np = np.array(theta_list) 
            np.savez(os.path.join(data_dir, filename), final_x=final_x_np, final_y=final_y_np, w=w_np, theta=theta_np)

    # plt.hist(w_list)
    # plt.show()

    # plt.hist(theta_list)
    # plt.show()

def extract_angle_and_t_max(filename:str) -> tuple[float, float]:
    # angle
    angle = float(filename.split('\\')[-3].replace('deg', ''))
    t_max = float(filename.split('\\')[-1].replace('s.npz', ''))
    return angle, t_max

def analyze_data() -> None:
    import glob, numpy as np, matplotlib.pyplot as plt
    # for root, dirs, files in os.walk(BASE_PATH):
    #     for file in files:
    #         print(os.path.join(BASE_PATH, file))
    
    files = glob.glob(os.path.join(BASE_PATH, '**/*.npz'), recursive=True)
    angles = np.linspace(0, 2*np.pi, 37)
    t_maxs = np.linspace(10, 200, 20)
    X, Y = np.meshgrid(angles, t_maxs)
    Z = np.zeros_like(X)

    ts = []
    thetas = []
    zs = []
    print(X.shape, Y.shape)
    for file in files:
        angle, t_max = extract_angle_and_t_max(file)

        data = np.load(file)
        x, y = data['final_x'], data['final_y']
        w, theta = data['w'], data['theta']

        idx_w = np.argmin(w), np.argmax(w)

        z = np.sqrt(np.cov(x)**2 + np.cov(y)**2)
        # theta_mean = np.mean(theta)

        ts.append(t_max)
        thetas.append(angle)
        zs.append(z)
        print(angle, t_max, z)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(x, y, c='r')
        ax.scatter([x[idx] for idx in idx_w], [y[idx] for idx in idx_w], c='b', marker='x')
        ax.set_xlabel('Final x [m]]')
        ax.set_ylabel('Final y [m]')
        plt.show()
        plt.close()


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ts, thetas, zs, c=zs)
    ax.set_xlabel('Final time [s]')
    ax.set_ylabel('Angle [deg]')
    ax.set_zlabel('Sigma [m]')
    plt.show()



if __name__=="__main__":
    # generate_data()
    analyze_data()