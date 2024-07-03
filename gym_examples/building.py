import pygame
from gym_examples.constants import *

class Building():
    def __init__(self):
        self.IN = (0, SCREENHEIGHT - RECT_LENGTH, RECT_WIDTH, RECT_LENGTH)
        self.OUT = (SCREENWIDTH - RECT_WIDTH, SCREENHEIGHT - RECT_LENGTH, RECT_WIDTH, RECT_LENGTH)

    def render(self, screen):
        pygame.draw.rect(screen, IN_COLOR, self.IN)
        pygame.draw.rect(screen, OUT_COLOR, self.OUT)
