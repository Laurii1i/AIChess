# Third-party imports
import pygame as pg

#  Local package imports
from config import HEIGHT, WIDTH

class StartMenu:
    """
    Represents the start screen for the Chess GUI.

    This menu displays a title overlay and buttons for selecting 
    whether to play as white or black. The menu remains active 
    until the player clicks one of the start buttons.

    Attributes:
        chessGUI: Reference to the main Chess GUI object.
        screen (pygame.Surface): Display surface used for rendering.
        title_text (str): Title text displayed at the top of the screen.
        start_game_buttons (list): Buttons to start the game as white or black.
        title_font (pygame.Font): Font used to render the title.
        active (bool): Indicates whether the menu is currently active.
    """

    def __init__(self, chessGUI, title_text: str) -> None:
        """Initialize the StartMenu with title text and start buttons."""
        self.chessGUI = chessGUI
        self.screen = chessGUI.screen
        self.title_text = title_text
        self.start_game_buttons = chessGUI.initiate_start_game_buttons()
        self.title_font = pg.font.SysFont(None, 48)
        self.active = True

    def draw(self) -> None:
        """
        Render the start menu overlay and buttons on the screen.
        """
        self.chessGUI.board.draw()
        self.chessGUI.set_transparent_overlay()

        for button in self.start_game_buttons:
            button.draw_on_screen()

        title_surface = self.title_font.render(
            self.title_text, True, (255, 255, 255)
        )
        self.screen.blit(
            title_surface,
            (
                WIDTH // 2 - title_surface.get_width() // 2,
                HEIGHT // 2 - 100,
            ),
        )

        pg.display.flip()

    def initiate(self):
        """
        Display the start menu loop until the user clicks a start button.

        Returns:
            Any: The output from the pressed button's callback.
        """
        choosing = True
        while choosing:
            self.draw()
            output = self.chessGUI.track_button_press(self.start_game_buttons)
            if output:
                break
        return output
