import pygame
from gym_examples.utils.constants import *

class Building():
    def __init__(self):
        self.IN = (0, SCREENHEIGHT - RECT_LENGTH, RECT_WIDTH, RECT_LENGTH)
        self.OUT = (SCREENWIDTH - RECT_WIDTH, SCREENHEIGHT - RECT_LENGTH, RECT_WIDTH, RECT_LENGTH)
