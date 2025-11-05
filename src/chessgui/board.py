# Standard library
import chess

# Third-party imports
import pygame
import numpy as np

# Local imports
from .piece import Piece
from config import SQ_SIZE, DIMENSION, ROOT
import utils as ut


class Board(chess.Board):
     
    def __init__(self, chessGUI):
          
        super().__init__()
        self.WHITE = (240, 217, 181)
        self.BLACK = (181, 136, 99)
        self.colors = (self.WHITE, self.BLACK)
        self.screen = chessGUI.screen
        self.chessGUI = chessGUI

        self.dragging_piece = None
        self.drag_offset = (0,0)
        self.selected_sq = None
        self.played_bot_moves = 0
        self.create_pieces()

    def create_pieces(self):
        """
        Initialize all chess pieces (both white and black) in their standard starting positions.

        This method clears any existing pieces on the board, then calls
        `create_white_pieces()` and `create_black_pieces()` to populate
        the `self.pieces` list with new `Piece` objects.
        """
        self.pieces = []
        self.create_white_pieces()
        self.create_black_pieces()

    def create_white_pieces(self):
        """
        Create and position all white chess pieces for a new game.

        The back-rank pieces (R, N, B, Q, K) are placed on rank 1 (row index 7),
        following standard chess setup, with Rooks, Knights, and Bishops mirrored
        symmetrically across the board. Pawns are placed on rank 2 (row index 6).

        Implementation Details:
            - Uses `load_image(symbol)` to load piece sprites.
            - Creates mirrored counterparts for R, N, and B on both flanks.
            - Adds 8 pawns to the second rank.
        """
        piece_order = ('R', 'N', 'B', 'Q', 'K')  # Order of back-rank pieces (a1–e1)

        # Create back-rank pieces (mirrored for R, N, B)
        for i, symbol in enumerate(piece_order):
            img = self.load_image(symbol)

            # Left-side piece
            self.pieces.append(Piece(
                self.chessGUI, symbol=symbol, image=img,
                screen=self.screen, location=(i, 7)
            ))

            # Mirror right-side for R, N, B only
            if i < 3:
                self.pieces.append(Piece(
                    self.chessGUI, symbol=symbol, image=img,
                    screen=self.screen, location=(7 - i, 7)
                ))

        # Create white pawns
        pawn_img = self.load_image('P')
        for i in range(8):
            self.pieces.append(Piece(
                self.chessGUI, symbol='P', image=pawn_img,
                screen=self.screen, location=(i, 6)
            ))

    def create_black_pieces(self):
        """
        Create all black chess pieces by mirroring the existing white pieces.

        This method clones each white piece, flips its color, mirrors its position
        vertically (across the horizontal center of the board), and reuses cached
        images where possible for performance efficiency.

        Implementation Details:
            - Symbols are converted to lowercase for black.
            - Uses a local image cache (`loaded_imgs`) to avoid redundant file I/O.
            - Mirrors only the vertical coordinate: (col, row) → (col, 7 - row).

        Side Effects:
            - Appends 16 black `Piece` objects to `self.pieces`.

        Notes:
            - Assumes `create_white_pieces()` has already populated `self.pieces`
            with the white set before this method is called.
        """
        loaded_imgs = {}  # Cache to avoid reloading
        white_pieces_copy = list(self.pieces)  # Avoid modifying list during iteration

        for white_piece in white_pieces_copy:
            black_symbol = white_piece.symbol.lower()

            # Load or reuse cached image
            if black_symbol not in loaded_imgs:
                loaded_imgs[black_symbol] = self.load_image(black_symbol)

            img = loaded_imgs[black_symbol]

            # Mirror vertically (same column, opposite row)
            col, row = white_piece.location
            black_location = (col, 7 - row)

            self.pieces.append(Piece(
                self.chessGUI, symbol=black_symbol, image=img,
                screen=self.screen, location=black_location
            ))


    def load_image(self, symbol):
        """
        Load and scale a chess piece image based on its symbol.

        Args:
            symbol (str): Uppercase for white, lowercase for black (e.g., 'K', 'p').

        Returns:
            pygame.Surface: The loaded and scaled image of the piece.
        """
        color_prefix = 'W' if symbol.isupper() else 'B'
        file_path = ROOT / 'src' / 'images' / f"{color_prefix}{symbol.upper()}-Photoroom.png"
        return pygame.transform.scale(pygame.image.load(file_path), (SQ_SIZE, SQ_SIZE))

    def get_piece(self, symbol):
        return next((p for p in self.pieces if p.symbol == symbol), None)
    
    def draw_board(self):
        """
        Draw the chessboard squares on the Pygame screen.
        """
        colors = [self.WHITE, self.BLACK]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[(r + c) % 2]
                pygame.draw.rect(
                    self.screen, color,
                    pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE)
                )


    def draw_pieces(self, exclude=None):
        """
        Draw all alive and non-dragged pieces on the board.

        Args:
            exclude (Piece, optional): A piece to exclude from drawing (e.g., the one being dragged).
        """
        for piece in self.pieces:
            if piece.dragging or not piece.alive or piece is exclude:
                continue
            piece.draw()


    def draw_dragging_piece(self, x, y):
        """
        Draw the currently dragged piece at the given mouse coordinates.

        Args:
            x (int): X coordinate of the mouse.
            y (int): Y coordinate of the mouse.
        """
        self.screen.blit(self.dragging_piece.image, (x, y))


    def draw(self, exclude=None):
        """
        Draw the board and all pieces.

        Args:
            exclude (Piece, optional): A piece to exclude from drawing.
        """
        self.draw_board()
        self.draw_pieces(exclude)


    def update_dragging_piece_location(self):
        """
        Update the location of the currently dragged piece
        to follow the mouse cursor.
        """
        piece = self.dragging_piece
        mouse_x, mouse_y = pygame.mouse.get_pos()
        x = mouse_x - self.drag_offset[0]
        y = mouse_y - self.drag_offset[1]
        piece.location = (x, y)


    def get_promotion_choice_pieces(self, color):
        """
        Retrieve the four pieces available for pawn promotion.

        Args:
            color (str): 'white' or 'black'.

        Returns:
            list[Piece]: List of promotion option pieces (Q, R, B, N or lowercase variants).
        """
        symbols = ('Q', 'R', 'B', 'N') if color == 'white' else ('q', 'r', 'b', 'n')
        return [next(p for p in self.pieces if p.symbol == s) for s in symbols]


    def castle(self, move_uci, animate=False):
        """
        Perform a castling move for white or black.

        Args:
            move_uci (str): UCI string representing the king’s move (e.g., 'e1g1').
            animate (bool, optional): Whether to animate the movement.
        """
        white_king = next(p for p in self.pieces if p.symbol == 'K')
        black_king = next(p for p in self.pieces if p.symbol == 'k')

        if move_uci[-1] == '1':  # White castles
            if animate:
                white_king.movement_animation(move_uci[2:])
            white_king.move_to(move_uci)

            white_rook = next(
                (p for p in self.pieces if p.symbol == 'R' and self.columns_distance_between(white_king, p) < 3),
                None
            )
            target_sq = 'f1' if move_uci[-2] == 'g' else 'd1'
            white_rook.move_to(f'--{target_sq}')
            if animate:
                white_rook.movement_animation(target_sq)
        else:  # Black castles
            if animate:
                black_king.movement_animation(move_uci[2:])
            black_king.move_to(move_uci)

            black_rook = next(
                (p for p in self.pieces if p.symbol == 'r' and self.columns_distance_between(p, black_king) < 3),
                None
            )
            target_sq = 'f8' if move_uci[-2] == 'g' else 'd8'
            if animate:
                black_rook.movement_animation(target_sq)
            black_rook.move_to(f'--{target_sq}')


    def is_castle(self, piece, move_uci):
        """
        Check if a move corresponds to a castling move.

        Args:
            piece (Piece): The piece being moved.
            move_uci (str): The move in UCI format.

        Returns:
            bool: True if the move is a castle, False otherwise.
        """
        return piece.symbol.lower() == 'k' and move_uci in ('e1g1', 'e1c1', 'e8g8', 'e8c8')


    def capture_en_passant(self, uci):
        """
        Handle the special en passant capture.

        Args:
            uci (str): The move in UCI format (e.g., 'e5d6').
        """
        start_rank = int(uci[1])
        end_rank = int(uci[3])
        target_rank = 8 - min(start_rank, end_rank)
        target_file = ut.letter_to_index(uci[2])

        captured = next(
            (p for p in self.pieces if p.location == (target_file, target_rank)), 
            None
        )
        if captured:
            captured.alive = False


    def reset_pieces(self):
        """
        Reset the board to the initial position and clear the piece list.
        """
        self.reset()
        self.pieces = []
        self.played_bot_moves = 0


    def invert_positions(self):
        """
        Invert all piece positions to flip the board view.
        """
        for piece in self.pieces:
            col, row = piece.location
            piece.location = (7 - col, 7 - row)


    def columns_distance_between(self, piece1, piece2):
        """
        Calculate the horizontal distance (in columns) between two pieces.

        Args:
            piece1 (Piece): First piece.
            piece2 (Piece): Second piece.

        Returns:
            int: The absolute difference in column indices.
        """
        return np.abs(piece2.location[0] - piece1.location[0])


    def get_piece_with_sq(self, sq):
        """
        Retrieve the piece located on a given square (e.g., 'e4').

        Args:
            sq (str): The square identifier (e.g., 'e4').

        Returns:
            Piece: The piece located on that square.
        """
        column, row = ut.letter_to_index(sq[0]), 8 - int(sq[1])

        if self.inverted:
            column, row = 7 - column, 7 - row

        return next(p for p in self.pieces if p.location == (column, row))
