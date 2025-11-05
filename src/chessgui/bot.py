# Third-party imports
import torch
import chess

# Local imports
from model import ChessNet
import utils as ut
from mappings import uci_index_connector
from config import ROOT

class Bot:
    """
    Chess AI Bot that interacts with a GUI and selects moves using a neural network.

    Attributes:
        chessGUI: The chess GUI interface containing the current board state.
        nn: The neural network used to evaluate board positions and select moves.
    """

    def __init__(self, chessGUI):
        """
        Initialize the Bot with a chess GUI and load pretrained model weights.

        Args:
            chessGUI: An instance of the GUI class that holds the chess board.
        """
        self.chessGUI = chessGUI
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nn = ChessNet()
        self.nn.load_state_dict(
            torch.load(ROOT / 'src' / 'model' / 'model_weights.pth', map_location = self.device)
        )
        self.nn.eval()

    def make_move(self) -> None:
        """
        Select and execute a move for the bot on the current board.

        The bot uses a stochastic policy for the first 4 moves and switches
        to a deterministic policy afterward. Handles castling, piece movement,
        and promotion animations, then updates the board state.
        """
        board: chess.Board = self.chessGUI.board
        stochastic = board.played_bot_moves < 4 # The first 4 rounds the bot chooses the move stochastically from the three best choices. This was done to add some randomness to the early game for more variety in gameplay

        fen = board.fen()
        move = self.get_bot_move_with_fen(fen, stochastic=stochastic)

        piece = board.get_piece_with_sq(move[:2])

        if board.is_castle(piece, move):
            board.castle(move, animate=True)
        else:
            from_sq, to_sq = move[:2], move[2:]
            for piece in self.chessGUI.board.pieces:
                if piece.is_at_square(from_sq) and piece.alive:
                    piece.movement_animation(to_sq)
                    piece.move_to(move)
                    if len(to_sq) == 3:
                        piece.promote(to_sq[-1])
                    break

        board.push_uci(move)
        board.played_bot_moves += 1

    def get_bot_move_with_fen(self, fen: str, stochastic: bool = False) -> str:
        """
        Get the bot's move for a given FEN position.

        Args:
            fen: A FEN string representing the current board state.
            stochastic: If True, select a move probabilistically from the top 3
                        moves; otherwise, choose the highest-scoring move.

        Returns:
            A UCI string representing the chosen move.
        """
        input_tensor = ut.fen_to_tensor(fen).unsqueeze(0)
        legal_mask = ut.get_mask(fen, uci_index_connector)

        with torch.no_grad():
            logits = self.nn(input_tensor)[0]
            legal_mask = legal_mask.bool()
            logits[0, ~legal_mask] = float('-inf')

            if stochastic:
                move_idx = self.sample_move_idx(logits)
            else:
                move_idx = torch.argmax(logits).item()

        return uci_index_connector[move_idx]

    def sample_move_idx(self, logits: torch.Tensor) -> int:
        """
        Sample a move index from the top-k moves based on their logits.

        Args:
            logits: A tensor of move logits from the neural network.

        Returns:
            An integer index corresponding to the chosen move.
        """
        k = 3
        topk = torch.topk(logits, k)
        topk_values = topk.values
        topk_indices = topk.indices

        probs = torch.softmax(topk_values, dim=1)
        chosen_index_in_topk = torch.multinomial(probs, 1).item()

        # Map back to the original index
        move_idx = topk_indices[0][chosen_index_in_topk].item()
        return move_idx