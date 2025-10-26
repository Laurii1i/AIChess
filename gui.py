# pygame_chess_bot.py
import pygame
import chess
import sys
from mappings import two_way

# ---------------------------
# Settings
# ---------------------------
WIDTH, HEIGHT = 700,700
DIMENSION = 8  # 8x8 board
SQ_SIZE = HEIGHT // DIMENSION
FPS = 60

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (186, 202, 68)

# Piece images dictionary
IMAGES = {}

tunnel = {
    # White pieces
    'P': 'WP',  # White Pawn
    'R': 'WR',  # White Rook
    'N': 'WN',  # White Knight
    'B': 'WB',  # White Bishop
    'Q': 'WQ',  # White Queen
    'K': 'WK',  # White King

    # Black pieces
    'p': 'BP',  # Black Pawn
    'r': 'BR',  # Black Rook
    'n': 'BN',  # Black Knight
    'b': 'BB',  # Black Bishop
    'q': 'BQ',  # Black Queen
    'k': 'BK',  # Black King
}

def load_images():
    pieces = ["WP","WR","WN","WB","WQ","WK","BP","BR","BN","BB","BQ","BK"]
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(pygame.image.load(f"images/{piece}.png"), (SQ_SIZE, SQ_SIZE))
    # You need to have an images folder with standard chess PNGs

# ---------------------------
# Pygame Board Class
# ---------------------------
class ChessGUI:
    def __init__(self, network_move_func):
        """
        network_move_func: function(board) -> chess.Move
        """
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess Bot GUI")
        self.clock = pygame.time.Clock()
        self.board = chess.Board()
        self.selected_sq = None
        self.dragging_piece = None
        self.drag_offset = (0,0)
        self.network_move_func = network_move_func
        load_images()

    def draw_board(self):
        colors = [WHITE, BLACK]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[(r+c)%2]
                pygame.draw.rect(self.screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_pieces(self):
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                sq = chess.square(c, 7-r)  # map row,col to chess square
                piece = self.board.piece_at(sq)
                if piece:
                    piece_name = piece.symbol()
                    piece_name = piece_name.upper() if piece.color == chess.WHITE else piece_name.lower()
                    key = tunnel[piece_name]
                    if self.dragging_piece and self.selected_sq == sq:
                        continue  # skip dragging piece
                    self.screen.blit(IMAGES[key], (c*SQ_SIZE, r*SQ_SIZE))

    def run(self):
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
                elif event.type == pygame.MOUSEMOTION:
                    self.on_mouse_motion(event.pos)

            self.draw_board()
            self.draw_pieces()
            if self.dragging_piece:
                x, y = pygame.mouse.get_pos()
                self.screen.blit(IMAGES[tunnel[self.dragging_piece]], (x - self.drag_offset[0], y - self.drag_offset[1]))
            pygame.display.flip()

    def on_mouse_down(self, pos):
        col = pos[0] // SQ_SIZE
        row = 7 - (pos[1] // SQ_SIZE)
        sq = chess.square(col, row)
        piece = self.board.piece_at(sq)
        if piece and piece.color == chess.WHITE:  # human plays white
            self.selected_sq = sq
            self.dragging_piece = piece.symbol().upper()
            offset_x = pos[0] - col*SQ_SIZE
            offset_y = pos[1] - (7-row)*SQ_SIZE
            self.drag_offset = (offset_x, offset_y)

    def on_mouse_motion(self, pos):
        pass  # handled in draw_pieces for dragging effect

    def on_mouse_up(self, pos):
        if self.selected_sq is None:
            return
        col = pos[0] // SQ_SIZE
        row = 7 - (pos[1] // SQ_SIZE)
        target_sq = chess.square(col, row)
        move = chess.Move(self.selected_sq, target_sq)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.selected_sq = None
            self.dragging_piece = None
            pygame.display.flip()
            pygame.time.delay(200)
            self.bot_move()
        else:
            # illegal move, reset drag
            self.selected_sq = None
            self.dragging_piece = None

    def bot_move(self):
        if self.board.is_game_over():
            return
        move = self.network_move_func(self.board)
        self.board.push_uci(move)

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    import torch
    import utils as ut  # your utilities: fen_to_tensor, get_mask, index_to_move
    from model2 import ChessNet  # your loaded model
    import time
    network = ChessNet()
    network.load_state_dict(torch.load(r"C:\Users\lauri\Documents\Chess\AIChess\improved_model1_groups_0-14.pth"))
    network.eval()  

    def network_move_func(board):
        # wrapper to call your model
        import torch
        fen = board.fen()
        input_tensor = ut.fen_to_tensor(fen).unsqueeze(0)
        legal_mask = ut.get_mask(fen)
        with torch.no_grad():
            logits = network(input_tensor)[0]
            legal_mask = legal_mask.bool()  # shape: (1968,)

            # Set logits to -inf where illegal
            logits[0, ~legal_mask] = float('-inf')
            move_idx = torch.argmax(logits).item()
        move = two_way[move_idx]
        return move

    gui = ChessGUI(network_move_func)
    gui.run()
