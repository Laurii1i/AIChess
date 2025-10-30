import chess
import pygame
from Piece import Piece
from config import SQ_SIZE, DIMENSION
import utils as ut
import numpy as np


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
        self.pieces = []
        self.played_bot_moves = 0
        #self.create_pieces()



    def create_pieces(self):

        self.pieces = []
        self.create_pawns()
        self.create_bishops()
        self.create_knights()
        self.create_rooks()
        self.create_queens()
        self.create_kings()

    def create_pawns(self):

        white_pawn_img = pygame.transform.scale(pygame.image.load(f"images/WP-Photoroom.png"), (SQ_SIZE, SQ_SIZE))
        black_pawn_img = pygame.transform.scale(pygame.image.load(f"images/BP-Photoroom.png"), (SQ_SIZE, SQ_SIZE))

        white_pawn_locations = [(i, 6) for i in range(0, 8)]
        black_pawn_locations = [(i, 1) for i in range(0, 8)]

        for loc in white_pawn_locations:
            self.pieces.append(Piece(chessGUI = self.chessGUI, symbol = 'P', image = white_pawn_img, screen = self.screen, location = loc))

        for loc in black_pawn_locations:
            self.pieces.append(Piece(chessGUI = self.chessGUI,symbol = 'p',image = black_pawn_img, screen = self.screen, location = loc))

    def create_bishops(self):

        
        white_bishop_img = pygame.transform.scale(pygame.image.load(f"images/WB-Photoroom.png"), (SQ_SIZE, SQ_SIZE))
        black_bishop_img = pygame.transform.scale(pygame.image.load(f"images/BB-Photoroom.png"), (SQ_SIZE, SQ_SIZE))

        white_bishop_locations = [(2,7), (5,7)]
        black_bishop_locations = [(2,0), (5,0)]

        for loc in white_bishop_locations:
            self.pieces.append(Piece(chessGUI = self.chessGUI,symbol = 'B',image = white_bishop_img, screen = self.screen, location = loc))

        for loc in black_bishop_locations:
            self.pieces.append(Piece(chessGUI = self.chessGUI,symbol = 'b',image = black_bishop_img, screen = self.screen, location = loc))

    def create_knights(self):

        
        white_knight_img = pygame.transform.scale(pygame.image.load(f"images/WN-Photoroom.png"), (SQ_SIZE, SQ_SIZE))
        black_knight_img = pygame.transform.scale(pygame.image.load(f"images/BN-Photoroom.png"), (SQ_SIZE, SQ_SIZE))

        white_knight_locations = [(1,7), (6,7)]
        black_knight_locations = [(1,0), (6,0)]

        for loc in white_knight_locations:
            self.pieces.append(Piece(chessGUI = self.chessGUI,symbol = 'N',image = white_knight_img, screen = self.screen, location = loc))

        for loc in black_knight_locations:
            self.pieces.append(Piece(chessGUI = self.chessGUI,symbol = 'n',image = black_knight_img, screen = self.screen, location = loc))

    def create_rooks(self):

        
        white_rook_img = pygame.transform.scale(pygame.image.load(f"images/WR-Photoroom.png"), (SQ_SIZE, SQ_SIZE))
        black_rook_img = pygame.transform.scale(pygame.image.load(f"images/BR-Photoroom.png"), (SQ_SIZE, SQ_SIZE))

        white_rook_locations = [(0,7), (7,7)]
        black_rook_locations = [(0,0), (7,0)]

        for loc in white_rook_locations:
            self.pieces.append(Piece(chessGUI = self.chessGUI,symbol = 'R',image = white_rook_img, screen = self.screen, location = loc))

        for loc in black_rook_locations:
            self.pieces.append(Piece(chessGUI = self.chessGUI,symbol = 'r',image = black_rook_img, screen = self.screen, location = loc))

    def create_queens(self):

        
        white_queen_img = pygame.transform.scale(pygame.image.load(f"images/WQ-Photoroom.png"), (SQ_SIZE, SQ_SIZE))
        black_queen_img = pygame.transform.scale(pygame.image.load(f"images/BQ-Photoroom.png"), (SQ_SIZE, SQ_SIZE))

        self.pieces.append(Piece(chessGUI = self.chessGUI,symbol = 'Q',image = white_queen_img, screen = self.screen, location = (3,7)))
        self.pieces.append(Piece(chessGUI = self.chessGUI,symbol = 'q',image = black_queen_img, screen = self.screen, location = (3,0)))

    
    def create_kings(self):

        
        white_king_img = pygame.transform.scale(pygame.image.load(f"images/WK-Photoroom.png"), (SQ_SIZE, SQ_SIZE))
        black_king_img = pygame.transform.scale(pygame.image.load(f"images/BK-Photoroom.png"), (SQ_SIZE, SQ_SIZE))

        self.pieces.append(Piece(chessGUI = self.chessGUI,symbol = 'K',image = white_king_img, screen = self.screen, location = (4,7)))
        self.pieces.append(Piece(chessGUI = self.chessGUI,symbol = 'k',image = black_king_img, screen = self.screen, location = (4,0)))
        


    def draw_board(self):
        colors = [self.WHITE, self.BLACK]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[(r+c)%2]
                pygame.draw.rect(self.screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_pieces(self, exclude = None):

        for piece in self.pieces:
            if piece.dragging or not piece.alive or piece is exclude:
                continue
            piece.draw()

    def draw_dragging_piece(self, x, y):

        self.screen.blit(self.dragging_piece.image, (x, y))

    def draw(self, exclude = None):
        self.draw_board()
        self.draw_pieces(exclude)

    def update_dragging_piece_location(self):

        piece = self.dragging_piece

        mouse_x, mouse_y = pygame.mouse.get_pos()
        x = mouse_x - self.drag_offset[0]
        y = mouse_y - self.drag_offset[1]

        piece.location = (x,y)

    def get_promotion_choice_pieces(self, color):

        pieces = []
        symbols = ('Q', 'R', 'B', 'N') if color == 'white' else ('q', 'r', 'b', 'n')
            
        for symbol in symbols:
            pieces.append(next(p for p in self.pieces if p.symbol == symbol))

        return pieces
    
    def castle(self, move_uci, animate = False):

        WK = next(piece for piece in self.pieces if piece.symbol == 'K')
        BK = next(piece for piece in self.pieces if piece.symbol == 'k')

        if move_uci[-1] == '1':

            if animate: WK.movement_animation(move_uci[2:])
            WK.move_to(move_uci)
            WR = next(
                (p for p in self.pieces if p.symbol == 'R' and self.columns_distance_between(WK, p) < 3),
                None)
            
            target_sq = 'f1' if move_uci[-2] == 'g' else 'd1'

            WR.move_to(f'--{target_sq}')
            if animate: WR.movement_animation(target_sq)
                

        else:

            if animate: BK.movement_animation(move_uci[2:])
            BK.move_to(move_uci)
            BR = next(
                (p for p in self.pieces if p.symbol == 'r' and self.columns_distance_between(p, BK) < 3),
                None)

            target_sq = 'f8' if move_uci[-2] == 'g' else 'd8'

            if animate: BR.movement_animation(target_sq)
            BR.move_to(f'--{target_sq}')

    def is_castle(self, piece, move_uci):

        return piece.symbol.lower() =='k' and move_uci in ('e1g1', 'e1c1', 'e8g8', 'e8c8')
    
    def capture_en_passant(self, uci):

        first_row, second_row = int(uci[1]), int(uci[3])

        if second_row > first_row: # White is en passanting
            target_row = 8 - first_row
            target_col_letter = uci[2]
            target_column = ut.letter_to_index(target_col_letter)

        else:
            target_row = 8 - second_row
            target_col_letter = uci[2]
            target_column = ut.letter_to_index(target_col_letter)

        deleted_piece = next((piece for piece in self.pieces if piece.location == (target_column, target_row)), None)
        deleted_piece.alive = False


    def reset_pieces(self):

        self.reset()
        self.pieces = []
        self.played_bot_moves = 0

    def invert_positions(self):

        for piece in self.pieces:

            col, row = piece.location
            new_col =  7 - col
            new_row = 7 - row
            piece.location = (new_col, new_row)

    def columns_distance_between(self, piece1, piece2):

        return np.abs(piece2.location[0]- piece1.location[0])
    
    def get_piece_with_sq(self, sq):

        column, row = ut.letter_to_index(sq[0]), 8 - int(sq[1])

        if self.inverted:
            column, row = 7 - column, 7 - row

        return next(piece for piece in self.pieces if piece.location == (column, row))