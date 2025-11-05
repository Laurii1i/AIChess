import webbrowser
import pygame as pg
import sys

"""
This file was originally generated with the help of ChatGPT.

It implements a simple instructional popup that appears when the GUI is launched.  
While the implementation works as intended, it’s not fully optimized — improvements can be made later if needed.
"""

class InstructionPopup:
    """
    Popup window displayed at game startup explaining how to play.
    Includes a short instruction text, a clickable link, and an OK button to close it.
    """

    def __init__(self, chessGUI, text, link):
        self.chessGUI = chessGUI
        self.screen = chessGUI.screen
        self.clock = chessGUI.clock
        self.text = text
        self.link = link

        # Fonts
        self.font = pg.font.SysFont(None, 25)
        self.title_font = pg.font.SysFont(None, 40, bold=True)
        self.ok_font = pg.font.SysFont(None, 32)
        self.link_font = pg.font.SysFont(None, 28)
        self.link_font.set_underline(True)

        # Layout
        self.ok_button_rect = pg.Rect(
            (self.screen.get_width() // 2 - 60,
             self.screen.get_height() - 120,
             120, 50)
        )

        # Colors
        self.bg_color = (20, 20, 20)
        self.text_color = (240, 240, 240)
        self.button_color = (60, 150, 60)
        self.button_hover = (80, 180, 80)
        self.link_color = (100, 160, 255)
        self.link_hover = (150, 200, 255)

        # Internal
        self.link_rect = None  # Will be set when drawing

    def draw_text(self):
        """Render wrapped instructional text inside the popup."""
        lines = self.wrap_text(self.text, 60)
        y = 120

        # Title
        title = self.title_font.render("How to Play", True, (255, 255, 255))
        self.screen.blit(title, (self.screen.get_width() // 2 - title.get_width() // 2, 50))

        # Body text
        for line in lines:
            txt_surface = self.font.render(line, True, self.text_color)
            self.screen.blit(txt_surface, (80, y))
            y += 30

        # Draw clickable link
        link_surface = self.link_font.render("Rules of chess (click to open)", True, self.link_color)
        link_pos = (80, y + 20)
        self.screen.blit(link_surface, link_pos)
        self.link_rect = link_surface.get_rect(topleft=link_pos)

    def draw_button(self, hover=False):
        """Draw the OK button."""
        color = self.button_hover if hover else self.button_color
        pg.draw.rect(self.screen, color, self.ok_button_rect, border_radius=8)
        text_surface = self.ok_font.render("OK", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.ok_button_rect.center)
        self.screen.blit(text_surface, text_rect)

    def run(self):
        """Main popup loop — waits for OK click or link click."""
        running = True
        while running:
            self.screen.fill(self.bg_color)
            self.draw_text()

            mouse_pos = pg.mouse.get_pos()
            hover_ok = self.ok_button_rect.collidepoint(mouse_pos)
            hover_link = self.link_rect and self.link_rect.collidepoint(mouse_pos)

            # Highlight link on hover
            if hover_link:
                link_surface = self.link_font.render("Rules of chess (click to open)", True, self.link_hover)
                self.screen.blit(link_surface, self.link_rect.topleft)

            self.draw_button(hover_ok)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if hover_ok:
                        running = False  # Close popup
                    elif hover_link:
                        webbrowser.open(self.link)

            pg.display.flip()
            self.clock.tick(30)

    @staticmethod
    def wrap_text(text, width):
        """Word-wrap text to fit the popup width."""
        words = text.split()
        lines, current = [], []
        for word in words:
            current.append(word)
            if len(" ".join(current)) > width:
                lines.append(" ".join(current[:-1]))
                current = [word]
        if current:
            lines.append(" ".join(current))
        return lines