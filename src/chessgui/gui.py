"""Chess GUI Module.

This module provides the `Gui` class, which manages the Pygame interface
for a chess bot game. It handles rendering, user interactions, and game logic
integration with the chess engine and board representation.
"""

# Standard library imports
import sys

# Third-party imports
import pygame
import chess

# Local package imports
from .bot import Bot
from .button import Button, ButtonText
from .startmenu import StartMenu
from .board import Board
from .instructionpopup import InstructionPopup
from config import HEIGHT, SQ_SIZE, WIDTH, FPS


class Gui:
    """Graphical User Interface for the Chess Bot."""

    def __init__(self):
        """Initialize the chess GUI and its components."""
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess Bot GUI")

        self.clock = pygame.time.Clock()
        self.board = Board(self)
        self.Bot = Bot(chessGUI=self)
        self.human_color = "white"  # Draw from white's perspective at start
        self.dragging_piece = None

        self.show_instructions_popup()
        self.startmenu = StartMenu(self, title_text="Let's play a game of chess!")

    # --- Menu ---

    def show_instructions_popup(self):
        popup_text = (
        "Welcome to AI chess bot!\n"
        "To play, drag and drop pieces to make moves.\n"
        "Your piece is returned if you try to do an illegal move!\n"
        "Castling and en-passant are fully supported.\n"
        "To castle, drag your king two squares to the right, or left.\n"
        "The bot will respond automatically after your move."
    )
        popup = InstructionPopup(self, popup_text, "https://www.instructables.com/Playing-Chess/")
        popup.run()

    def initiate_game(self, color: str) -> None:
        """Start a new game as the specified color.

        Args:
            color (str): 'white' or 'black' representing the player's color.
        """
        self.board.reset()
        self.human_color = color

        if self.human_color == "black":
            self.board.inverted = True
            self.board.create_pieces()
            self.board.invert_positions()
            self.Bot.make_move()
        else:
            self.board.inverted = False
            self.board.create_pieces()

        return 1
    def initiate_start_game_buttons(self) -> tuple[Button, Button]:
        """Create buttons for choosing player color at the start menu.

        Returns:
            tuple[Button, Button]: Buttons for 'Play as White' and 'Play as Black'.
        """
        height, width = 50, 200

        play_as_white = Button(
            ChessGUI=self,
            left=WIDTH // 4 - 100,
            top=HEIGHT // 2 + 50,
            width=width,
            height=height,
            color=(255, 255, 255),
            ButtonText=ButtonText(
                text="Play as White",
                color=(0, 0, 0),
                fontsize=36,
            ),
            function=lambda: self.initiate_game("white"),
        )

        play_as_black = Button(
            ChessGUI=self,
            left=3 * WIDTH // 4 - 100,
            top=HEIGHT // 2 + 50,
            width=width,
            height=height,
            color=(0, 0, 0),
            ButtonText=ButtonText(
                text="Play as Black",
                color=(255, 255, 255),
                fontsize=36,
            ),
            function=lambda: self.initiate_game("black"),
        )

        return play_as_white, play_as_black

    # --- Main Loop ---

    def run(self) -> None:
        """Main loop of the game — handles events, updates, and drawing."""
        self.startmenu.initiate()
        running = True

        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.on_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.on_mouse_up(event.pos)

            self.board.draw()

            if self.board.dragging_piece:
                x = event.pos[0] - self.board.drag_offset[0]
                y = event.pos[1] - self.board.drag_offset[1]
                self.board.dragging_piece.draw(location=(x, y))

            pygame.display.flip()

    # --- Mouse ---

    def on_mouse_down(self, pos: tuple[int, int]) -> None:
        """Handle mouse press event.

        Args:
            pos (tuple[int, int]): Screen coordinates of the mouse click.
        """
        dragging_piece = self.get_selected_piece(pos)
        if not dragging_piece: return

        if dragging_piece and dragging_piece.color == self.human_color:
            dragging_piece.dragging = True
            self.board.dragging_piece = dragging_piece
            self.board.drag_offset = (
                pos[0] - self.board.dragging_piece.location[0] * SQ_SIZE,
                pos[1] - self.board.dragging_piece.location[1] * SQ_SIZE,
            )

        if self.board.inverted:
            # Board is inverted (player is black) → flip x and y accordingly
            self.board.from_square = chess.square(
                7 - dragging_piece.location[0], dragging_piece.location[1]
            )
        else:
            # Normal orientation (player is white)
            self.board.from_square = chess.square(
                dragging_piece.location[0], 7 - dragging_piece.location[1]
            )

    def square_to_screen(self, location: tuple[int, int]) -> tuple[int, int]:
        """Convert board coordinates to screen pixel coordinates."""
        row, col = location
        if self.human_color == chess.WHITE:
            x = col * SQ_SIZE
            y = (7 - row) * SQ_SIZE
        else:
            x = (7 - col) * SQ_SIZE
            y = row * SQ_SIZE
        return x, y

    def get_selected_piece(self, pos: tuple[int, int]):
        """Get the chess piece at a given screen position."""
        col, row = self.screen_to_square(pos)
        for piece in self.board.pieces:
            if piece.location == (col, row) and piece.alive:
                return piece
        return None

    def on_mouse_up(self, pos: tuple[int, int]) -> None:
        """Handle mouse release event."""
        piece = self.board.dragging_piece
        if piece is None:
            return

        col, row = self.screen_to_square(pos)
        if self.board.inverted:
            col, row = 7 - col, 7 - row

        target_sq = chess.square(col, 7 - row)
        move = chess.Move(self.board.from_square, target_sq)

        piece.dragging = False

        # Handle promotion
        if piece.symbol.lower() == "p" and chess.square_rank(target_sq) in [0, 7]:
            promotion_symbol = self.open_promotion_popup(piece)
            move = chess.Move.from_uci(move.uci() + promotion_symbol)

        if move in self.board.legal_moves:

            self.make_move(piece, move)

            if not self.board.is_game_over():

                self.Bot.make_move()

                if self.board.is_game_over():
                    self.handle_game_over(player_won=False)
            else:
                self.handle_game_over(player_won=True)

        self.board.dragging_piece = None

    def handle_game_over(self, player_won: bool) -> None:
        """Handle the end-of-game screen and display the result."""
        if self.board.is_stalemate():
            self.startmenu.title_text = "It's a stalemate!"
        else:
            self.startmenu.title_text = (
                "Checkmate, you won!" if player_won else "Checkmate, you lost!"
            )
        self.startmenu.initiate()

    # --- Promotion Overlay ---
    def make_move(self, piece, move: chess.Move) -> None:
        move_uci = move.uci()

        if self.board.is_castle(piece, move_uci):
            self.board.castle(move_uci)
        else:
            en_passant: bool = self.board.is_en_passant(move)
            self.board.dragging_piece.move_to(move_uci, en_passant=en_passant)
        self.board.push_uci(move_uci)


    def open_promotion_popup(self, promoting_piece):
        """Open a popup overlay for pawn promotion selection."""
        promotion_choice_pieces = self.board.get_promotion_choice_pieces(
            color=self.human_color
        )

        promotion_buttons = self.create_promotion_buttons_from_pieces(
            pieces = promotion_choice_pieces,
            promoting_piece = promoting_piece
            )

        self.set_transparent_overlay()

        while True:
            self.clock.tick(FPS)
            self.draw_buttons(promotion_buttons)
            promotion_symbol = self.track_button_press(promotion_buttons)

            if promotion_symbol:
                break

        return promotion_symbol

    def create_promotion_buttons_from_pieces(self, pieces, promoting_piece):

        promotion_buttons = []

        for i, piece in enumerate(pieces, start=2):
            promotion_buttons.append(
                Button(
                    self,
                    left=i * SQ_SIZE,
                    top=2 * SQ_SIZE,
                    width=SQ_SIZE,
                    height=SQ_SIZE,
                    image=piece.image,
                    color=None,
                    ButtonText=None,
                    function=lambda p=piece: promoting_piece.promote(p.symbol),
                )
            )

        return promotion_buttons
    
    def draw_buttons(self, buttons: list[Button]):
        for button in buttons:
            button.draw_on_screen()
        pygame.display.flip()

    def track_button_press(self, buttons: list[Button]):
        """Scans for MOUSEBUTTONDOWN events on button objects from pygame.event generator. Returns None if buttons are not pressed.

            Args:
        buttons (list[Button]): A list of `Button` instances to check 
            for mouse collisions.

        Returns:
            Any | None: The return value of the pressed button's 
            `on_button_press()` function, or ``None`` if no button was pressed."""
        
        output = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for button in buttons:
                    if button.collidepoint(event.pos):
                        output = button.on_button_press()
        return output

    def set_transparent_overlay(self) -> None:
        """Create a semi-transparent overlay for modal popups."""
        surface = pygame.Surface((WIDTH, HEIGHT))
        surface.set_alpha(150)
        surface.fill((0, 0, 0))
        self.screen.blit(surface, (0, 0))

    def screen_to_square(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Convert screen coordinates to board square indices."""
        x, y = pos
        row = y // SQ_SIZE
        col = x // SQ_SIZE
        return col, row
