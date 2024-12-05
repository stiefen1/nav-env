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


SCREEN_SIZE = (600, 450)
SCREEN_COLOR = (100, 200, 255)
FPS = 60

# TODO: Regarder ce qu'on peut faire avec pygame.transform (https://www.pygame.org/docs/ref/transform.html) pour changer de référentiel / scale

pygame.init()

class PyGameScreen:
    def __init__(self, obstacles: ObstacleCollection):
        self.drawable = obstacles
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
            # self.draw()

            # Create a new surface to draw on
            draw_surface = pygame.Surface(SCREEN_SIZE)
            draw_surface.fill(SCREEN_COLOR)

            # Translate the origin to the center of the screen
            translated_surface = pygame.Surface(SCREEN_SIZE)
            translated_surface.fill(SCREEN_COLOR)
            translated_surface.blit(draw_surface, (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2))

            # Flip the y-axis
            flipped_surface = pygame.transform.flip(translated_surface, False, True)

            # Apply additional transformations (e.g., scaling, rotating)
            scaled_surface = pygame.transform.scale(flipped_surface, SCREEN_SIZE)
            rotated_surface = pygame.transform.rotate(scaled_surface, 180)  # Example rotation angle

            # Draw the objects onto the transformed surface
            self.drawable.draw(rotated_surface)

            # Blit the transformed surface onto the main screen
            self.screen.blit(rotated_surface, (0, 0))

            pygame.display.flip()
            self.clock.tick(FPS)
            pygame.display.update()
        pygame.quit()

    def draw(self):
        # val = 200-min([pygame.time.get_ticks()/100., 100])
        # print(val)
        # pygame.draw.polygon(self.screen, (255, 0, 0), [(100, 100), (200, 100), (200, val), (100, 200)])
        self.drawable.draw(self.screen)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.drawable})"
    
    def __str__(self):
        return self.__repr__()
    
    def __call__(self):
        self.run()

def test():
    from nav_env.obstacles.obstacles import Circle, Ellipse
    from nav_env.obstacles.obstacles import Obstacle
    import numpy as np

    Ncircle = 30 # Number of obstacles to generate
    Npoly = 30 # Number of obstacles to generate
    lim = ((0, 0), (800, 600))
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

    obstacles.append(Ellipse(100, 100, 50, 10))
    # obstacles.append(Ellipse(-100, 100, 10, 50))
    # obstacles.append(Ellipse(100, -100, 50, 10))
    # obstacles.append(Ellipse(-100, -100, 10, 50))

    viz = PyGameScreen(obstacles)
    viz.run()

if __name__ == "__main__":
    test()




