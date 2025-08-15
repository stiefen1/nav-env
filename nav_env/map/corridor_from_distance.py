import numpy as np
import networkx as nx
from nav_env.obstacles.obstacles import Obstacle, Circle

def test() -> None:
    
    import matplotlib.pyplot as plt
 
    o1 = Obstacle(xy=[(-2, -2), (2, -2), (2, 2), (-2, 2)]).rotate(30).translate(1, 3)
    o2 = Circle(0, 0, 3).translate(-3, -2)
    o3 = Obstacle(xy=[(-2, -3), (2, -2), (3, 2), (-2, 2)]).rotate(-30).translate(3, -3)
    o4 = Circle(0, 0, 3).translate(7, 2)
    o5 = Obstacle(xy=[(-2, -3), (2, -2), (3, 2), (-2, 2)]).rotate(-30).translate(-5, -8)
    obstacles = [o1, o2, o3, o4, o5]

    xlim = (-12, 12)
    ylim = (-12, 12)
    n = (200, 200)
    x = np.linspace(*xlim, int(n[0]))
    y = np.linspace(*ylim, int(n[1]))
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            cost = cost_fn(xi, yj, obstacles, tau=0.3)
            Z[j, i] = cost

    _, ax = plt.subplots()
    # for obs in obstacles:
    #     obs.plot(ax=ax)

    cont = ax.contourf(X, Y, Z, levels=[0.25, 0.6])
    plt.colorbar(cont, ticks=np.linspace(np.min(Z[:, :]), np.max(Z[:, :]), 5), ax=ax)
    plt.tight_layout()
    plt.show()
    

def cost_fn(x:float, y:float, obstacles:list[Obstacle], tau:float=1) -> float:
    cost = 0
    for obs in obstacles:
        cost += np.exp(-obs.distance((x, y))/tau)
    return cost



if __name__ == "__main__":
    test()