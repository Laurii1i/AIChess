# Third-party imports
import numpy as np
import pygame
import chess

# Local application imports
from config import SQ_SIZE, FPS, ROOT
import utils as ut

class Piece:

    def __init__(
            self,
            chessGUI: "Gui",
            symbol: str,
            image: pygame.Surface,
            screen: pygame.Surface,
            location: tuple[int, int],
        ):

        
        """
        Initialize a Piece instance.

        Args:
            chessGUI: The main chess GUI controller (used for board access and redraws).
            symbol (str): The piece's symbol (e.g., 'P', 'k').
            image (pygame.Surface): The image representing this piece.
            screen (pygame.Surface): The pygame screen where pieces are drawn.
            location (tuple[int, int]): The (column, row) position on the board.
        """
        self.chessGUI = chessGUI
        self.image = image
        self.symbol = symbol
        self.screen = screen
        self.location = location
        self.alive = True
        self.dragging = False
        self.color = 'white' if symbol.isupper() else 'black'

    def draw(self, location=None):
        """
        Draw the piece at its current board location or a temporary pixel position.

        Args:
            location (tuple[int, int], optional): Pixel coordinates to draw at.
                If None, draws at the board square corresponding to self.location.
        """
        if location is None:
            x = self.location[0] * SQ_SIZE
            y = self.location[1] * SQ_SIZE
            self.screen.blit(self.image, (x, y))
        else:
            self.screen.blit(self.image, location)

    def move_to(self, uci, en_passant=False):
        """
        Move this piece according to a UCI move string (e.g., 'e2e4').

        Args:
            uci (str): UCI-formatted move string.
            en_passant (bool): Whether the move is an en-passant capture.
        """
        board = self.chessGUI.board
        to_sq = uci[2:4]

        col, row = ut.letter_to_index(to_sq[0]), 8 - int(to_sq[1])
        if board.inverted:
            col, row = 7 - col, 7 - row

        # Capture any opposing piece at destination
        for piece in board.pieces:
            if piece.location == (col, row):
                piece.alive = False

        if en_passant:
            board.capture_en_passant(uci)

        self.location = (col, row)

        # Promotion (if applicable)
        if len(uci) == 5:
            self.promote(uci[-1])

    def movement_animation(self, to_sq):
        """
        Animate piece movement from current to target square.

        Args:
            to_sq (str): Destination square in algebraic notation (e.g., 'e4').
        """
        board = self.chessGUI.board
        col, row = ut.letter_to_index(to_sq[0]), 8 - int(to_sq[1])
        if board.inverted:
            col, row = 7 - col, 7 - row

        start_loc = np.array(self.location) * SQ_SIZE
        end_loc = np.array((col, row)) * SQ_SIZE

        frames = 20
        vector = end_loc - start_loc
        increment = vector / frames

        for i in range(1, frames + 1):
            draw_loc = start_loc + i * increment
            board.draw(exclude=self)
            self.draw(location=draw_loc.astype(int))
            self.chessGUI.clock.tick(FPS)
            pygame.display.flip()

    def is_at_square(self, square):
        """
        Check if this piece is located at the given algebraic square.

        Args:
            square (str): A square in algebraic notation (e.g., 'e4').

        Returns:
            bool: True if the piece occupies that square.
        """
        col = ut.letter_to_index(square[0])
        row = 8 - int(square[1])
        if self.chessGUI.board.inverted:
            col, row = 7 - col, 7 - row
        return self.location == (col, row)

    def is_white(self) -> bool:
        """Return True if this piece is white."""
        return self.symbol.isupper()

    def promote(self, symbol):
        """
        Promote this piece to another type (usually when reaching the end rank).

        Args:
            symbol (str): The new piece symbol (e.g., 'Q', 'N').

        Returns:
            str: The lowercase version of the new symbol.
        """
        color = 'W' if self.is_white() else 'B'
        path = ROOT / 'src' / f"images/{color}{symbol.upper()}-Photoroom.png"
        self.image = pygame.transform.scale(
            pygame.image.load(path), (SQ_SIZE, SQ_SIZE)
        )
        self.symbol = symbol.upper() if self.is_white() else symbol.lower()
        return symbol.lower()