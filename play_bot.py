import torch
import chess
import chess.engine
import chess.pgn
import numpy as np
import model
import utils as ut
from model import two_way
import time

network = model.ChessNet()
network.load_state_dict(torch.load(r"C:\Users\lauri\Documents\Chess\AIChess\training1_weights.pth"))
network.eval()  

def select_move(board):
    fen = board.fen()
    input = ut.fen_to_tensor(fen)
    legal_moves_mask = ut.get_mask(fen)
    with torch.no_grad():
        logits = network(input.unsqueeze(0))[0]  # shape (1, N_moves)
        logits = logits.masked_fill(legal_moves_mask == 0, float('-inf'))  
    best_move_idx = torch.argmax(logits).item()
    uci = two_way[best_move_idx]
    return uci

board = chess.Board()

while not board.is_game_over():
    

    if board.turn == chess.WHITE:
        move = input("Your move (e.g. e2e4): ")
        
        try:
            board.push_uci(move)
        except:
            print("Invalid move.")
            continue
        time.sleep(1)
        print(board)
    else:
        move = select_move(board)
        print(f"Bot plays: {move}")
        time.sleep(1)
        board.push_uci(move)
        print(board)