from config import SQ_SIZE, FPS
import numpy as np
import pygame

class Piece:

    def __init__(self, chessGUI, symbol, image, screen, location):
        
        self.chessGUI = chessGUI
        self.image = image
        self.symbol = symbol
        self.screen = screen
        self.location = location
        self.alive = True
        self.dragging = False
        self.color = 'white' if symbol.isupper() else 'black'
        self.letter_to_col = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
            'h': 7
        }

    def draw(self, location = None):

        if location is None:

            x = self.location[0] * SQ_SIZE
            y = self.location[1] * SQ_SIZE
            self.screen.blit(self.image, (x,y))
        else:
            self.screen.blit(self.image, location)

    def move_to(self, uci, en_passant = False):

        board = self.chessGUI.board
        to_sq = uci[2:]
        to_sq = to_sq[:2]

        col, row = self.letter_to_col[to_sq[0]], 8 - int(to_sq[1])
        if board.inverted:
            col, row = 7 - col, 7 - row

        for p in self.chessGUI.board.pieces:
            if p.location == (col, row):
                p.alive = False
        
        if en_passant:
            board.capture_en_passant(uci)

        self.location = (col, row)
        if len(to_sq) == 3:

            self.promote(uci)

    def movement_animation(self, to_sq):
    
        board = self.chessGUI.board
        to_sq = to_sq[:2]
        to_sq = self.letter_to_col[to_sq[0]], (8 - int(to_sq[1]))
        start_loc = np.array(self.location) * SQ_SIZE

        if board.inverted:
            to_sq = tuple(map(lambda x: 7 - x, to_sq))

        end_loc = np.array(to_sq) * SQ_SIZE

        frames = 20
        vector = end_loc - start_loc

        increment = vector / frames

        
        for i in range(1, frames + 1):

            draw_loc = start_loc + i * increment
            self.chessGUI.board.draw(exclude = self)            
            self.draw(location=draw_loc.astype(int))
            self.chessGUI.clock.tick(FPS)
            pygame.display.flip()

        
    def is_at_square(self, square):

        check_location = self.letter_to_col[square[0]], (8 - int(square[1]))

        if self.chessGUI.board.inverted:
            check_location = tuple(map(lambda x: 7 - x, check_location))
        return check_location == self.location
            
    def is_white(self) -> bool:

        return self.symbol.isupper()

    def promote(self, symbol):

        color = 'W' if self.is_white() else 'B'
        self.image = pygame.transform.scale(pygame.image.load(f"images/{color}{symbol}-Photoroom.png"), (SQ_SIZE, SQ_SIZE))
        self.symbol = symbol.upper() if self.is_white() else symbol.lower()

        return symbol.lower()

