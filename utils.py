
import numpy as np
import chess
import torch
from mappings import two_way
import matplotlib.pyplot as plt
import math

# Map chess pieces to plane indices

def white_to_play(board) -> bool:

    if board.turn == chess.WHITE:
        return True
    else:
        return False


def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Convert a FEN string to a [12, 8, 8] tensor.
    Planes 0-5: White pawn, knight, bishop, rook, queen, king
    Planes 6-11: Black pawn, knight, bishop, rook, queen, king
    """
    piece_to_plane = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
                    }
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = piece_to_plane[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6  # Shift for black pieces

            # Convert chess.SQUARE (0=a1..63=h8) to row/col in tensor
            row = 7 - (square // 8)  # rank 8 = row 0
            col = square % 8
            tensor[plane, row, col] = 1.0

    tensor = torch.from_numpy(tensor)

    turn_plane = torch.ones((8, 8), dtype=tensor.dtype) if board.turn == chess.WHITE else torch.zeros((8, 8), dtype=tensor.dtype)
    tensor = torch.cat([tensor, turn_plane.unsqueeze(0)], dim=0)  # shape: (13, 8, 8)

    return tensor

PROMOTION_PLANES_PER_SQUARE = 12  # 3 directions * 4 promotion pieces

def uci_to_index(board: chess.Board, uci: str) -> int:
    """
    Convert a UCI move to a fixed move index in AlphaZero-style 4672 move space.
    """
    move = chess.Move.from_uci(uci)
    from_sq = move.from_square
    to_sq = move.to_square

    # ---- Normal moves (non-promotion) ----
    if move.promotion is None:
        return from_sq * 64 + to_sq

    # ---- Promotion moves ----
    # Determine direction: 0=forward, 1=left capture, 2=right capture
    file_diff = chess.square_file(to_sq) - chess.square_file(from_sq)
    direction = {0:0, -1:1, 1:2}[file_diff]

    # Promotion piece mapping: QUEEN=0, ROOK=1, BISHOP=2, KNIGHT=3
    promotion_map = {chess.QUEEN:0, chess.ROOK:1, chess.BISHOP:2, chess.KNIGHT:3}
    piece_index = promotion_map[move.promotion]

    plane_index = direction * 4 + piece_index

    # Determine color and offset
    rank = chess.square_rank(from_sq)
    file_index = chess.square_file(from_sq)  # 0..7
    if rank == 6:  # White promotion on rank 7
        return 4096 + file_index * PROMOTION_PLANES_PER_SQUARE + plane_index
    elif rank == 1:  # Black promotion on rank 2
        return 4096 + 8 * PROMOTION_PLANES_PER_SQUARE + file_index * PROMOTION_PLANES_PER_SQUARE + plane_index
    else:
        raise ValueError(f"Invalid promotion move from square {from_sq}")


def index_to_uci(board: chess.Board, index: int) -> str:
    """
    Convert a flattened AlphaZero-style move index back to a UCI move string.
    """
    # ---- Normal moves ----
    if index < 4096:
        from_sq = index // 64
        to_sq = index % 64
        return chess.Move(from_sq, to_sq).uci()

    # ---- Promotion moves ----
    promotion_offset = index - 4096
    if promotion_offset < 8 * PROMOTION_PLANES_PER_SQUARE:  # White promotions
        from_file = promotion_offset // PROMOTION_PLANES_PER_SQUARE
        plane_index = promotion_offset % PROMOTION_PLANES_PER_SQUARE
        from_sq = chess.square(from_file, 6)  # rank 7
        to_rank_delta = 1
    else:  # Black promotions
        promotion_offset -= 8 * PROMOTION_PLANES_PER_SQUARE
        from_file = promotion_offset // PROMOTION_PLANES_PER_SQUARE
        plane_index = promotion_offset % PROMOTION_PLANES_PER_SQUARE
        from_sq = chess.square(from_file, 1)  # rank 2
        to_rank_delta = -1

    direction = plane_index // 4  # 0=forward,1=left,2=right
    piece_index = plane_index % 4
    promotion_map = {0:chess.QUEEN,1:chess.ROOK,2:chess.BISHOP,3:chess.KNIGHT}
    promotion_piece = promotion_map[piece_index]

    # Compute to-square
    file = chess.square_file(from_sq)
    if direction == 0:  # forward
        to_file = file
    elif direction == 1:  # capture-left
        to_file = file - 1
    else:  # capture-right
        to_file = file + 1

    to_sq = chess.square(to_file, chess.square_rank(from_sq) + to_rank_delta)

    move = chess.Move(from_sq, to_sq, promotion=promotion_piece)
    return move.uci()


def line_to_best_move(row):

    fen, line = row['fen'], row['line']
    board = chess.Board(fen)
    space_idx = line.find(' ')
    uci = line[:space_idx]
    legal_ucis = {move.uci() for move in board.legal_moves}
    if uci in legal_ucis:
        return uci
    else:
        fromS, toS = uci[:2], uci[2:]
        if toS == 'h8':
            toS = 'g8'
        elif toS in ('a8', 'b8'):
            toS = 'c8'
        elif toS in ('a1', 'b1'):
            toS ='c1'
        elif toS == 'h1':
            toS = 'g1'
        else:
            pass

        if fromS + toS not in legal_ucis:
            return 'nan'

        return fromS + toS


def cp_to_value_target(cp, clip=600.0):
    """
    Convert centipawn to scalar target in [-1, 1] using tanh normalization.
    cp: tensor of shape (B,)
    clip: clip value in centipawns (typical 600 - 1000)
    """
    # scale to roughly [-1, 1], but use tanh for smooth tail behaviour
    scaled = cp / clip
    result = math.tanh(scaled)  # returns values in (-1,1)
    return result

def get_mask(fen):

    board = chess.Board(fen)
    moves = [two_way[i.uci()] for i in list(board.legal_moves)]    
    
    mask = torch.zeros(1968, dtype=torch.bool)  # start with all zeros
    mask[moves] = 1
    return mask

def plot(losses, line_raw, line_smooth, ax, window=10):
    """
    Update the loss plot dynamically.
    - losses: list of batch losses
    - line_raw, line_smooth: Line2D objects from ax.plot()
    - ax: matplotlib axes
    - window: moving average window
    """
    if len(losses) == 0:
        return

    x = np.arange(len(losses))
    
    if len(losses) < window:
        smoothed = np.array(losses)
    else:
        smoothed = np.convolve(losses, np.ones(window) / window, mode='same')

    line_raw.set_data(x, losses)
    line_smooth.set_data(x, smoothed)

    ax.relim()
    ax.autoscale_view()

    plt.pause(0.001)

def letter_to_index(letter):
    return ord(letter.lower()) - ord('a')
