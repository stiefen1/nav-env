"""
Ideal usage:
    - Create a class that inherits from BaseViz
    - Implement the abstract methods
    - Use the class to visualize the environment
"""
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from nav_env.obstacles.collection import ObstacleCollection
from nav_env.obstacles.obstacles import Obstacle, Circle
from nav_env.environment.environment import NavigationEnvironment 


SCREEN_SIZE = (1200, 900)
SCREEN_COLOR = (100, 200, 255)
FPS = 60

# TODO: Regarder ce qu'on peut faire avec pygame.transform (https://www.pygame.org/docs/ref/transform.html) pour changer de référentiel / scale

pygame.init()

class PyGameScreen:
    def __init__(self, env:NavigationEnvironment, *args, scale:float=1, **kwargs):
        self.env = env
        self._scale = scale
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()
        self.done = False
        self.value = 0

    def run(self):
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        self.value = max(0, self.value-1)
                        self.drawable.group_obstacles_closer_than(self.value)
                    elif event.key == pygame.K_UP:
                        self.value += 1
                        self.drawable.group_obstacles_closer_than(self.value)

            self.screen.fill(SCREEN_COLOR)
            self.env.draw()
            
        pygame.quit()

    def play(self,
             t0:float=0,
             tf:float=10,
             dt:float=0.03,
             own_ships_verbose=['enveloppe', 'frame', 'acceleration', 'velocity', 'forces'],
             target_ships_verbose=['enveloppe'],
             **kwargs
             ):
        """
        Play the environment during an interval of time.
        """
        t = t0
        pause = False
        done = False
        while t < tf and not done:
            start_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        pause = not pause

            self.screen.fill(SCREEN_COLOR)
            
            if not pause:
                self.env.step()
                t += dt

            self.env.draw(t, self.screen, own_ship_physics=own_ships_verbose, target_ship_physics=target_ships_verbose, scale=self._scale, **kwargs)
            pygame.display.flip()
            
            elapsed_time = (pygame.time.get_ticks() - start_time) / 1000
            if elapsed_time < dt:
                pygame.time.wait(int((dt - elapsed_time) * 1000))
                
            print(t-t%1)
            self.clock.tick(FPS)

        pygame.quit()
    
    def __repr__(self):
        return f"{self.__class__.__name__}"
    
    def __str__(self):
        return self.__repr__()
    
    def __call__(self):
        self.run()

def test_old():
    from nav_env.obstacles.obstacles import Circle, Ellipse
    from nav_env.obstacles.obstacles import Obstacle
    import numpy as np

    Ncircle = 30 # Number of obstacles to generate
    Npoly = 30 # Number of obstacles to generate
    lim = ((-SCREEN_SIZE[0]/2, -SCREEN_SIZE[1]/2), (SCREEN_SIZE[0]/2, SCREEN_SIZE[1]/2))
    xmin, ymin = lim[0]
    xmax, ymax = lim[1]
    rmin, rmax = 1, 20
    obsmin, obsmax = -30, 30
    np.random.seed(0)

    obstacles = ObstacleCollection()
    for _ in range(Ncircle):
        c = Circle(np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax), np.random.uniform(rmin, rmax))
        obstacles.append(c)

    for _ in range(Npoly):
        xy = np.random.uniform(obsmin, obsmax, (8, 2))
        o = Obstacle(xy=xy).convex_hull().translate(np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))
        obstacles.append(o)

    obstacles.append(Ellipse(100, 100, 100, 10))
    obstacles.append(Ellipse(-100, 100, 10, 50))
    obstacles.append(Ellipse(100, -100, 50, 10))
    obstacles.append(Ellipse(-100, -100, 10, 50))

    viz = PyGameScreen(obstacles)
    viz.run()


def test():
    from nav_env.obstacles.obstacles import ObstacleWithKinematics
    from nav_env.obstacles.collection import ObstacleWithKinematicsCollection
    from nav_env.ships.sailing_ship import SailingShip
    o1 = ObstacleWithKinematics(lambda t: (t, -t, t*10), xy=[(0, 0), (2, 0), (2, 2), (0, 2)]).rotate(45).translate(0., 9.)
    o2 = ObstacleWithKinematics(lambda t: (t, t, t*20), xy=[(0, 0), (2, 0), (2, 2), (0, 2)]).rotate(45).translate(0., 0.)
    ts1 = SailingShip(length=20, width=10, ratio=7/9, p0=(-10, 10, 0), v0=(1, -1, 0), make_heading_consistent=True)
    coll = ObstacleWithKinematicsCollection([o1, o2, ts1])

    env = NavigationEnvironment(obstacles=coll)
    screen = PyGameScreen(env)
    screen.play()

if __name__ == "__main__":
    test()




