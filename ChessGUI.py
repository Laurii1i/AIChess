# pygame_chess_bot_promotion_overlay.py
import pygame
import chess
import sys
from Bot import Bot
from Button import Button, ButtonText
from StartMenu import StartMenu
from Board import Board
from config import HEIGHT, SQ_SIZE, WIDTH, FPS
# ---------------------------
# Settings
# ---------------------------


# Colors


# Piece images dictionary

# ---------------------------
# Chess GUI
# ---------------------------




class ChessGUI:
    def __init__(self):

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess Bot GUI")
        self.clock = pygame.time.Clock()
        self.board = Board(self)
        self.Bot = Bot(chessGUI = self)
        self.human_color = 'white' # We use whites perspective to draw pieces in the start screen        
        self.dragging_piece = None
        self.startmenu = StartMenu(self, title_text = "Let's play a game of chess!")

    # --- Menu ---
    def initiate_game(self, color):
 
        self.board.reset()
        self.human_color = color
        if self.human_color == 'black':
            self.board.inverted = True
            self.board.create_pieces()
            self.board.invert_positions()
            self.Bot.make_move()
            
        else:
            self.board.inverted = False
            self.board.create_pieces()

    def initiate_start_game_buttons(self) -> Button:

        height, width = 50, 200
        play_as_white = Button(ChessGUI = self,
                               left = WIDTH//4 - 100,
                               top = HEIGHT//2 + 50, 
                               width = width, 
                               height = height, 
                               color = (255,255,255), 
                               ButtonText = ButtonText(text = 
                                                 'Play as White', 
                                                 color = (0,0,0), 
                                                 fontsize = 36), 
                               function = lambda: self.initiate_game('white'))
        
        play_as_black = Button(ChessGUI = self,
                               left = 3 * WIDTH//4 - 100, 
                               top = HEIGHT//2 + 50, 
                               width = width, 
                               height = height, 
                               color = (0,0,0), 
                               ButtonText = ButtonText(text = 'Play as Black', 
                                                 color = (255, 255, 255), 
                                                 fontsize = 36),
                                function = lambda: self.initiate_game('black'))

        return play_as_white, play_as_black
    

    # --- Board ---
   

    # --- Main Loop ---
    def run(self):
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
                y = event.pos[1]- self.board.drag_offset[1]
                self.board.dragging_piece.draw(location = (x, y))

            pygame.display.flip()

    # --- Mouse ---
    def on_mouse_down(self, pos):
        
        dragging_piece = self.get_selected_piece(pos)
        if dragging_piece and dragging_piece.color == self.human_color:
            dragging_piece.dragging = True
            self.board.dragging_piece = dragging_piece
            self.board.drag_offset = (pos[0] - self.board.dragging_piece.location[0] * SQ_SIZE, pos[1] - self.board.dragging_piece.location[1] * SQ_SIZE)

            if not self.board.inverted:
                self.board.from_square = chess.square(dragging_piece.location[0], 7 - dragging_piece.location[1])
            else:
                self.board.from_square = chess.square(7 - dragging_piece.location[0], dragging_piece.location[1])

    def square_to_screen(self, location):

        row, col = location
        if self.human_color == chess.WHITE:
            x = col * SQ_SIZE
            y = (7 - row) * SQ_SIZE
        else:
            x = (7 - col) * SQ_SIZE
            y = row * SQ_SIZE
        return x, y

    def get_selected_piece(self, pos):
        
        col, row = self.screen_to_square(pos)

        for p in self.board.pieces:

            if p.location == (col, row) and p.alive:
                return p
        else:
            return None
        
    def on_mouse_up(self, pos):

        piece = self.board.dragging_piece
        if piece is None:
            return
        
        col, row = self.screen_to_square(pos)

        if self.board.inverted:
            col, row = 7 - col, 7 - row

        target_sq = chess.square(col, 7 - row)
        move = chess.Move(self.board.from_square, target_sq)

        # Handle promotion
        
        piece.dragging = False
        if piece and piece.symbol.lower() == 'p' and chess.square_rank(target_sq) in [0,7]:
            promotion_symbol = self.open_promotion_popup(piece)
            move = chess.Move.from_uci(move.uci() + promotion_symbol)

        if move in self.board.legal_moves:

            move_uci = move.uci()  
            
            if self.board.is_castle(piece, move_uci):
                self.board.castle(move_uci)  # push to board
            else:
                en_passant: bool = self.board.is_en_passant(move)
                self.board.dragging_piece.move_to(move_uci, en_passant = en_passant)
            self.board.push_uci(move_uci)

            if not self.board.is_game_over():
                self.Bot.make_move()           
                if self.board.is_game_over():
                    self.startmenu.title_text = "It's a stalemate!" if self.board.is_stalemate() else 'Checkmate, you lost!'     
                    self.startmenu.initiate()           
            else:
                self.startmenu.title_text = "It's a stalemate!" if self.board.is_stalemate() else 'Checkmate, you won!' 
                self.startmenu.initiate()

        self.board.dragging_piece = None

    # --- Promotion Overlay ---
    def open_promotion_popup(self, promoting_piece):

        promotion_choice_pieces = self.board.get_promotion_choice_pieces(color = self.human_color)
        promotion_choices = []

        for i, piece in enumerate(promotion_choice_pieces, start = 2):
            promotion_choices.append(Button(self,
                                            left = i * SQ_SIZE, 
                                            top = 2*SQ_SIZE, 
                                            width = SQ_SIZE, 
                                            height = SQ_SIZE, 
                                            image = piece.image, 
                                            color = None, 
                                            ButtonText = None, 
                                            function = lambda p = piece: promoting_piece.promote(p.symbol)))

        choosing = True
        self.set_transparent_overlay()
        while choosing:

            self.clock.tick(FPS)
            for choice in promotion_choices:
                choice.draw_on_screen()

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for choice in promotion_choices:

                        if choice.collidepoint(event.pos):

                            promotion_symbol = choice.on_button_press()
                            choosing = False

        return promotion_symbol

    def set_transparent_overlay(self):
        s = pygame.Surface((WIDTH, HEIGHT))
        s.set_alpha(150)
        s.fill((0, 0, 0))
        self.screen.blit(s, (0, 0))


    def screen_to_square(self, pos):
        x, y = pos

        row = y // SQ_SIZE
        col = x // SQ_SIZE

        return col, row



