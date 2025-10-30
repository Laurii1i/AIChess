import pygame
from dataclasses import dataclass

@dataclass
class ButtonText:
    text: str
    color: tuple
    fontsize: int

class Button(pygame.Rect):
    def __init__(self, ChessGUI, left, top, width, height, color: tuple, ButtonText: ButtonText, function, image = None):
        # Call the parent class (pygame.Rect) constructor
        super().__init__(left, top, width, height, border_radius=10)

        
        self.screen = ChessGUI.screen
        self.function = function
        self.left, self.top = left, top

        if image is None:
            self.type = 'Textbox'
            self.buttontext = ButtonText
            self.color = color
            self.font = pygame.font.SysFont(None, ButtonText.fontsize)
            self.text_surface = self.font.render(self.buttontext.text, True, self.buttontext.color)
            self.text_rect = self.text_surface.get_rect(center=self.center)
        else:
            self.type = 'Image'
            self.image = image

        pygame.display.flip()

    def draw_on_screen(self):

        if self.type == 'Textbox':
            pygame.draw.rect(self.screen, self.color, self)
            self.text_rect.center = self.center
            self.screen.blit(self.text_surface, self.text_rect)
        else:
            self.screen.blit(self.image, (self.left, self.top))

    def on_button_press(self):

        return self.function()