import math
import numpy as np
import torch
import chess
import matplotlib.pyplot as plt


def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Convert a FEN string into a (13, 8, 8) tensor representation.

    The tensor encodes:
      - Planes 0–5: white pieces (pawn, knight, bishop, rook, queen, king)
      - Planes 6–11: black pieces (pawn, knight, bishop, rook, queen, king)
      - Plane 12: current player's turn (1 for white, 0 for black)

    Args:
        fen (str): FEN string representing a board position.

    Returns:
        torch.Tensor: Tensor of shape (13, 8, 8) encoding board state and turn.
    """
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = piece_to_plane[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6  # Shift to black planes

            # Convert square index (0=a1 ... 63=h8) to tensor row/col
            row = 7 - (square // 8)  # rank 8 → row 0
            col = square % 8
            tensor[plane, row, col] = 1.0

    tensor = torch.from_numpy(tensor)

    turn_plane = (
        torch.ones((8, 8), dtype=tensor.dtype)
        if board.turn == chess.WHITE
        else torch.zeros((8, 8), dtype=tensor.dtype)
    )

    return torch.cat([tensor, turn_plane.unsqueeze(0)], dim=0)


def line_to_best_move(row: dict) -> str:
    """
    Extract the first move (in UCI format) from a move line string,
    correcting common errors and ensuring legality.

    Args:
        row (dict): Row containing 'fen' (FEN string) and 'line' (move sequence).

    Returns:
        str: The first legal UCI move, or 'nan' if none found.
    """
    fen, line = row["fen"], row["line"]
    board = chess.Board(fen)

    space_idx = line.find(" ")
    uci = line if space_idx == -1 else line[:space_idx]

    legal_ucis = {move.uci() for move in board.legal_moves}
    if uci in legal_ucis:
        return uci

    # Sometimes castling "to square" (where king moves) is marked "incorrectly" in the data,
    # this is due to chess960 (chess with randomized initial position), where castling works slightly differently.
    from_sq, to_sq = uci[:2], uci[2:]
    if to_sq == "h8":
        to_sq = "g8"
    elif to_sq in ("a8", "b8"):
        to_sq = "c8"
    elif to_sq in ("a1", "b1"):
        to_sq = "c1"
    elif to_sq == "h1":
        to_sq = "g1"

    return from_sq + to_sq if from_sq + to_sq in legal_ucis else "nan"


def cp_to_value_target(cp: float, clip: float = 600.0) -> float:
    """
    Convert centipawn evaluation to scalar in [-1, 1] using tanh normalization.

    Args:
        cp (float): Centipawn value (positive for white advantage).
        clip (float): Scaling factor to control normalization range.

    Returns:
        float: Normalized evaluation in (-1, 1).
    """
    scaled = cp / clip
    return math.tanh(scaled)


def get_mask(fen: str, uci_index_connector: dict) -> torch.Tensor:
    """
    Create a legal-move mask tensor for the given board position.

    Args:
        fen (str): FEN representation of the board.
        uci_index_connector (dict): Mapping from move UCI strings to indices.

    Returns:
        torch.Tensor: Boolean tensor of shape (1968,) where legal moves = 1.
    """
    board = chess.Board(fen)
    moves = [uci_index_connector[m.uci()] for m in board.legal_moves]

    mask = torch.zeros(1968, dtype=torch.bool)
    mask[moves] = True
    return mask


def letter_to_index(letter: str) -> int:
    """
    Convert a file letter ('a'–'h') to a column index (0–7).

    Args:
        letter (str): File letter.

    Returns:
        int: Corresponding zero-based column index.
    """
    return ord(letter.lower()) - ord("a")
