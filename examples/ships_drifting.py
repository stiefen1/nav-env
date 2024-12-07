from nav_env.viz.matplotlib_screen import MatplotlibScreen as Screen
from nav_env.environment.environment import NavigationEnvironment as Env
from nav_env.ships.collection import ShipCollection
from nav_env.ships.ship import *
from nav_env.wind.wind_source import UniformWindSource
from nav_env.obstacles.obstacles import *

# Simulation parameters
lim = 300
xlim, ylim = (-lim, -lim), (lim, lim)
dt = 0.05
tf = 20

# Shore (Made of obstacles)
obs1 = Circle(0, 40, 50)
obs2 = Ellipse(-50, -50, 100, 20)
obs3 = Obstacle(xy=[(0, 0), (50, 0), (80, 10), (100, 50), (60, 90), (10, 30)]).rotate_and_translate(200, 30, 90)
shore = ObstacleCollection([obs1, obs2, obs3])

# Ships
ship1 = Ship(ShipStates3(-150., -200., 180., 0., 0., 30.), integrator=Euler(dt), name="Ship1")
ship2 = Ship(ShipStates3(-150., 50., -70., 10., 0., -10.), integrator=Euler(dt), name="Ship2")
ship3 = Ship(ShipStates3(10., -100., -30., 0., 0., 0.), integrator=Euler(dt), name="Ship3")
ship4 = Ship(ShipStates3(250., -200., 0., 0., 0., 60.), integrator=Euler(dt), name="Ship4")
ship5 = Ship(ShipStates3(250., 250., 80., -20., -20., 10.), integrator=Euler(dt), name="Ship5")
own_ships = ShipCollection([ship1, ship2])
target_ships = ShipCollection([ship3, ship4, ship5])

# Wind
uniform_wind = UniformWindSource(10, 45)

# Environment
env = Env(
    own_ships=own_ships,
    target_ships=target_ships,
    wind_source=uniform_wind,
    shore=shore
    )

# Screen to display simulation results
screen = Screen(env, scale=1, lim=(xlim, ylim))
screen.play(dt=dt, tf=tf)  