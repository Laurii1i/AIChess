import torch
from model2 import ChessNet
import chess
import utils as ut
from mappings import two_way
from config import SQ_SIZE

class Bot:

    def __init__(self, chessGUI):

        self.chessGUI = chessGUI
        self.NN = ChessNet()
        self.NN.load_state_dict(torch.load(r"C:\Users\lauri\Documents\Chess\AIChess\improved_model1_groups_0-14.pth"))
        self.NN.eval()

    def make_move(self) -> None:
        
        board: chess.Board = self.chessGUI.board

        stochastic = True if board.played_bot_moves < 4 else False

        fen = board.fen()
        move = self.get_bot_move_with_fen(fen, stochastic = stochastic)

        piece = board.get_piece_with_sq(move[:2])

        if board.is_castle(piece, move):
            board.castle(move, animate = True)
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

        input_tensor = ut.fen_to_tensor(fen).unsqueeze(0)
        legal_mask = ut.get_mask(fen)
        with torch.no_grad():
            logits = self.NN(input_tensor)[0]
            legal_mask = legal_mask.bool()
            logits[0, ~legal_mask] = float('-inf')

            if stochastic:
                move_idx = self.sample_move_idx(logits)
            else:
                #finite_mask = torch.isfinite(logits)
                #finite_logits = logits[finite_mask]
                #move_idx = torch.argmin(finite_logits).item()
                move_idx = torch.argmax(logits).item()

        move = two_way[move_idx]
        return move
    
    def sample_move_idx(self, logits):

        k = 3 
        topk = torch.topk(logits, k)
        topk_values = topk.values
        topk_indices = topk.indices
        probs = torch.softmax(topk_values, dim=1)
        chosen_index_in_topk = torch.multinomial(probs, 1).item()

        # Map it back to the original index
        move_idx = topk_indices[0][chosen_index_in_topk].item()
        return move_idx