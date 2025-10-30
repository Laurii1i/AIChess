# pygame_chess_bot_promotion_overlay.py
import pygame
import chess
import sys
from Bot import Bot
from Button import Button, ButtonText
from StartMenu import StartMenu
# ---------------------------
# Settings
# ---------------------------
WIDTH, HEIGHT = 600, 600
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
FPS = 60

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)

# Piece images dictionary
IMAGES = {}

tunnel = {
    'P': 'WP', 'R': 'WR', 'N': 'WN', 'B': 'WB', 'Q': 'WQ', 'K': 'WK',
    'p': 'BP', 'r': 'BR', 'n': 'BN', 'b': 'BB', 'q': 'BQ', 'k': 'BK',
}

def load_images():
    pieces = ["WP","WR","WN","WB","WQ","WK","BP","BR","BN","BB","BQ","BK"]
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(
            pygame.image.load(f"images/{piece}.png"), (SQ_SIZE, SQ_SIZE)
        )

# ---------------------------
# Chess GUI
# ---------------------------




class ChessGUI:
    def __init__(self):

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess Bot GUI")
        self.clock = pygame.time.Clock()
        self.board = chess.Board()
        self.Bot = Bot(chessGUI = self)
        self.human_color = chess.WHITE # We use whites perspective to draw pieces in the start screen        
        self.dragging_piece = None
        load_images()
        self.startmenu = StartMenu(self, title_text = "Let's play a game of chess!")
        #self.initiate_game()

    # --- Menu ---
    def initiate_game(self, color):

        self.board = chess.Board()
        self.selected_sq = None
        self.dragging_piece = None
        self.drag_offset = (0,0)        
        self.human_color = chess.WHITE if color == 'white' else chess.BLACK
        if self.human_color == chess.BLACK:
            self.Bot.make_move()

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
    def draw_board(self):
        colors = [WHITE, BLACK]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[(r+c)%2]
                pygame.draw.rect(self.screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_pieces(self):
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                if self.human_color == chess.WHITE:
                    sq = chess.square(c, 7-r)
                    draw_r, draw_c = r, c
                else:
                    sq = chess.square(7-c, r)
                    draw_r, draw_c = r, c
                piece = self.board.piece_at(sq)
                if piece:
                    key = tunnel[piece.symbol()]
                    if self.dragging_piece and self.selected_sq == sq:
                        continue
                    self.screen.blit(IMAGES[key], (draw_c*SQ_SIZE, draw_r*SQ_SIZE))

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

            self.draw_board()
            self.draw_pieces()
            if self.dragging_piece:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                draw_x = mouse_x - self.drag_offset[0]
                draw_y = mouse_y - self.drag_offset[1]
                self.screen.blit(IMAGES[tunnel[self.dragging_piece]], (draw_x, draw_y))
            pygame.display.flip()

    # --- Mouse ---
    def on_mouse_down(self, pos):
        col, row = self.screen_to_square(pos)
        sq = chess.square(col, row)
        piece = self.board.piece_at(sq)
        if piece and piece.color == self.human_color:
            self.selected_sq = sq
            self.dragging_piece = piece.symbol()
            sq_rect_x, sq_rect_y = self.square_to_screen(col, row)
            self.drag_offset = (pos[0] - sq_rect_x, pos[1] - sq_rect_y)

    def on_mouse_up(self, pos):
        if self.selected_sq is None:
            return
        col, row = self.screen_to_square(pos)
        target_sq = chess.square(col, row)
        move = chess.Move(self.selected_sq, target_sq)

        # Handle promotion
        piece = self.board.piece_at(self.selected_sq)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(target_sq) in [0,7]:
            move.promotion = self.choose_promotion()

        if move in self.board.legal_moves:
            # Convert move to UCI, including promotion
            move_uci = move.uci()  
            self.board.push_uci(move_uci)  # push to board

            self.selected_sq = None
            self.dragging_piece = None
            pygame.time.delay(200)

            # Ensure bot sees the correct board state
            if not self.board.is_game_over():
                self.Bot.make_move()           
                if self.board.is_game_over():
                    self.startmenu.initiate()
            else:
                self.startmenu.initiate()
        else:
            self.selected_sq = None
            self.dragging_piece = None

    # --- Promotion Overlay ---
    def choose_promotion(self):
        options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        piece_symbols = ['Q','R','B','N']

        panel_width = 4 * SQ_SIZE
        panel_height = SQ_SIZE
        panel_x = WIDTH//2 - panel_width//2
        panel_y = HEIGHT//2 - panel_height//2

        rects = [pygame.Rect(panel_x + i*SQ_SIZE, panel_y, SQ_SIZE, SQ_SIZE) for i in range(4)]

        choosing = True
        while choosing:
            # Only draw small panel behind pieces (no fullscreen overlay)
            pygame.draw.rect(self.screen, (200, 200, 200), (panel_x, panel_y, panel_width, panel_height))

            # Draw piece images
            for i, rect in enumerate(rects):
                color_prefix = 'W' if self.human_color == chess.WHITE else 'B'
                img_key = color_prefix + piece_symbols[i]
                self.screen.blit(IMAGES[img_key], (rect.x, rect.y))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for i, rect in enumerate(rects):
                        if rect.collidepoint(event.pos):
                            choosing = False
                            return options[i]


    def set_transparent_overlay(self):
        s = pygame.Surface((WIDTH, HEIGHT))
        s.set_alpha(150)
        s.fill((0, 0, 0))
        self.screen.blit(s, (0, 0))


    def screen_to_square(self, pos):
        x, y = pos
        if self.human_color == chess.WHITE:
            col = x // SQ_SIZE
            row = 7 - (y // SQ_SIZE)
        else:
            col = 7 - (x // SQ_SIZE)
            row = y // SQ_SIZE
        return col, row

    def square_to_screen(self, col, row):
        if self.human_color == chess.WHITE:
            x = col * SQ_SIZE
            y = (7 - row) * SQ_SIZE
        else:
            x = (7 - col) * SQ_SIZE
            y = row * SQ_SIZE
        return x, y

    # --- Bot ---

    def animate_move(self, move_uci):
        move = chess.Move.from_uci(move_uci)
        piece = self.board.piece_at(move.from_square)
        if not piece:
            return

        key = tunnel[piece.symbol()]

        start_col, start_row = chess.square_file(move.from_square), chess.square_rank(move.from_square)
        end_col, end_row = chess.square_file(move.to_square), chess.square_rank(move.to_square)

        if self.human_color == chess.WHITE:
            start_x, start_y = start_col * SQ_SIZE, (7 - start_row) * SQ_SIZE
            end_x, end_y = end_col * SQ_SIZE, (7 - end_row) * SQ_SIZE
        else:
            start_x, start_y = (7 - start_col) * SQ_SIZE, start_row * SQ_SIZE
            end_x, end_y = (7 - end_col) * SQ_SIZE, end_row * SQ_SIZE

        frames = 15
        for i in range(1, frames + 1):
            self.draw_board()
            # Draw all pieces except the moving one
            for r in range(DIMENSION):
                for c in range(DIMENSION):
                    if self.human_color == chess.WHITE:
                        sq = chess.square(c, 7-r)
                        draw_r, draw_c = r, c
                    else:
                        sq = chess.square(7-c, r)
                        draw_r, draw_c = r, c
                    p = self.board.piece_at(sq)
                    if p and sq != move.from_square:
                        img_key = tunnel[p.symbol()]
                        self.screen.blit(IMAGES[img_key], (draw_c*SQ_SIZE, draw_r*SQ_SIZE))

            # Draw the animated piece at interpolated position
            x = start_x + (end_x - start_x) * i / frames
            y = start_y + (end_y - start_y) * i / frames
            self.screen.blit(IMAGES[key], (x, y))

            pygame.display.flip()
            self.clock.tick(FPS)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    #network = ChessNet()
    #network.load_state_dict(torch.load(r"C:\Users\lauri\Documents\Chess\AIChess\improved_model1_groups_0-14.pth"))


    gui = ChessGUI()
    gui.run()
