from typing import List
import pygame
from gym_examples.utils.building import Building
from gym_examples.utils.constants import *
from gym_examples.utils.humans import Human

class SimulationVisualiser():
    def __init__(self, is_render=False):
        self.is_render = is_render
        if self.is_render:
            pygame.init()
            pygame.display.set_caption('HitL AC Simulation')
            self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
            self.font = pygame.font.SysFont("Arial", FONTSIZE)
            self.setBackground()
        else:
            self.screen = None
            self.font = None
    
    def setBackground(self):
        """
        Sets up the background surface for the Pygame display and fills it with a background color
        """
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BG_COLOR)
    
    def update(self, humans_in:List[Human], humans_out:List[Human], building: Building, curr_time, reward:int, 
               vote_up:int, vote_down:int, temp_setpt:int, time_interval, sample_size:int):
        if self.is_render:
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.background, (0, 0))

            self.render_building(building)
            for human_in in humans_in: self.render_human(human_in)
            for human_out in humans_out: self.render_human(human_out)

            # display stats
            def _displayText(description, pos_description, value, pos_value):
                text_description = self.font.render(description, True, BLACK)
                self.screen.blit(text_description, text_description.get_rect(center=pos_description))
                text_value = self.font.render(value, True, BLACK)
                self.screen.blit(text_value, text_value.get_rect(center=pos_value))

            _displayText("Total population of the day: ", 
                        (75, FONTSIZE),
                        str(sample_size), 
                        (165, FONTSIZE))
            _displayText("Time: ", 
                        (17, 3*FONTSIZE),
                        f"{curr_time - time_interval} ~ {curr_time}", 
                        (145, 3*FONTSIZE))
            _displayText("Votes (UP / DOWN): ", 
                        (57, 5*FONTSIZE),
                        str(vote_up) + " / " + str(vote_down), 
                        (140, 5*FONTSIZE))
            _displayText("HLAC RL (temperature / reward): ", 
                        (90, 7*FONTSIZE),
                        str(temp_setpt) + " / " + str(reward), 
                        (280, 7*FONTSIZE))
            
            pygame.display.update()
            return self._render_frame()
           
    def close(self):
        pygame.display.quit()
        pygame.quit()
    
    def render_building(self, building: Building):
        """Given a building object, draw rectangle to represent the in and out building"""
        pygame.draw.rect(self.screen, IN_COLOR, building.IN)
        pygame.draw.rect(self.screen, OUT_COLOR, building.OUT)

    def render_human(self, human:Human, is_inside:bool): #TODO: Move renderer out of the agent
        color = AGENT_IN_COLOR if is_inside else AGENT_OUT_COLOR
        pygame.draw.rect(self.screen, color, (human.x, human.y, TEXT_BOX_LENGTH, TEXT_BOX_HEIGHT))
        text = self.font.render(str(human.id), True, BLACK) if self.font is not None else None
        self.screen.blit(text, text.get_rect(center=(self.x + 15, self.y + FONTSIZE)))