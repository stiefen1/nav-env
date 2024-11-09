"""
Ideal usage:
    - Create a class that inherits from BaseViz
    - Implement the abstract methods
    - Use the class to visualize the environment
"""

import pygame

class BaseViz:
    def __init__(self, env):
        self.env = env
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.done = False

    def run(self):
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def draw(self):
        raise NotImplementedError