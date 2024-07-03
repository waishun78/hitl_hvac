import pygame
import numpy as np
from constants import *

class Agent(object):
    def __init__(self, id, color=AGENT_OUT_COLOR, data={}, x=0, y=0, font=None, screen=None):
        self.id = id
        self.color = color
        self.data = data
        self.x = x
        self.y = y
        self.radius = 10
        self.text = font.render(str(id), True, BLACK) if font is not None else None
        self.screen = screen

    def randomIn(self):
        self.color = AGENT_IN_COLOR
        self.x = np.random.rand() * (RECT_WIDTH - TEXT_BOX_LENGTH)
        self.y = np.random.rand() * (RECT_LENGTH - TEXT_BOX_HEIGHT) + (SCREENHEIGHT - RECT_LENGTH)
        return self

    def randomOut(self):
        self.color = AGENT_OUT_COLOR
        self.x = np.random.rand() * (RECT_WIDTH - TEXT_BOX_LENGTH) + (SCREENWIDTH - RECT_WIDTH)
        self.y = np.random.rand() * (RECT_LENGTH - TEXT_BOX_HEIGHT) + (SCREENHEIGHT - RECT_LENGTH)
        return self

    def updatePos(self, new_x, new_y):
        self.x = new_x
        self.y = new_y
        return self

    def render(self):
        pygame.draw.rect(self.screen, self.color, (self.x, self.y, TEXT_BOX_LENGTH, TEXT_BOX_HEIGHT))
        self.screen.blit(self.text, self.text.get_rect(center=(self.x + 15, self.y + FONTSIZE)))
