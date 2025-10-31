import pygame
import sys
from config import HEIGHT, WIDTH

class StartMenu:

    def __init__(self, chessGUI, title_text):
        
        self.chessGUI = chessGUI
        self.screen = chessGUI.screen
        self.title_text = title_text
        self.start_game_buttons = chessGUI.initiate_start_game_buttons()
        self.title_font = pygame.font.SysFont(None, 48)
        self.active = True
        

    def draw(self):

        self.chessGUI.board.draw()
        self.chessGUI.set_transparent_overlay()
        for but in self.start_game_buttons:
            but.draw_on_screen()

        self.title = self.title_font.render(self.title_text, True, (255, 255, 255))
        self.screen.blit(self.title, (WIDTH//2 - self.title.get_width()//2, HEIGHT//2 - 100))
        
        pygame.display.flip()

    def initiate(self):

        choosing = True
        while choosing:
            self.draw()            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:

                    for start_game_button in self.start_game_buttons:
                        if start_game_button.collidepoint(event.pos):
                            start_game_button.on_button_press()
                            choosing = False
                            break
        
        self.chessGUI.board.dragging_piece = None
