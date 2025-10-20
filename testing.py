import chess
import utils as ut
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
from model import ChessNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from mappings import two_way
import pyarrow as pa
import matplotlib.pyplot as plt
import os
# Create a new chess board
"""
board = chess.Board()        # standard starting position
board.reset()                # reset board to starting position
board.fen()                  # get FEN string of current position
board.turn                    # True if White's turn, False if Black's
board.fullmove_number         # counts full moves (starts at 1)
board.is_game_over()          # True if checkmate, stalemate, or draw"""

batch_size = 64
criterion = nn.CrossEntropyLoss()
board = chess.Board()

script_dir = Path(__file__).resolve().parent

file_path = script_dir / Path('../TrainingData/training_data_0-7.parquet')
parquet_file = pq.ParquetFile(file_path)
table = parquet_file.read_row_group(1, columns = ('fen', 'line'))
df = table.to_pandas()

#df = df.iloc[:200000].reset_index(drop=True)
df['best_move'] = df.apply(ut.line_to_best_move, axis = 1)
df = df[df['best_move'] != 'nan'].reset_index(drop=True)
df['tensor'] = df['fen'].apply(ut.fen_to_tensor) # 48 seconds for million rows
df['best_move_labels'] = df['best_move'].map(two_way)
df['legal_moves_mask'] = df['fen'].apply(ut.get_mask)

network = ChessNet()

all_tensors = torch.stack(df['tensor'].tolist())                 # shape: (num_rows, 12, 8, 8)
all_labels = torch.tensor(df['best_move_labels'].tolist(), dtype=torch.long)
all_masks  = torch.stack(df['legal_moves_mask'].tolist())        # shape: (num_rows, 1968)

dataset = TensorDataset(all_tensors, all_labels, all_masks)
loader = DataLoader(dataset, batch_size=64)

device = torch.device('cuda')
network = network.to(device)  # move model to GPU if available
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)

losses = []

plt.ion()
fig, ax = plt.subplots()
line_raw, = ax.plot([], [], 'b-', alpha=0.3, label='Batch Loss')    # raw batch loss
line_smooth, = ax.plot([], [], 'r-', label='Smoothed Loss')         # moving average
ax.set_xlabel("Batch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Evolution")
ax.legend()

#network.load_state_dict(torch.load(os.path.join(script_dir,"training1_weights.pth")))
#network.eval()  

for batch_tensors, batch_labels, batch_masks in loader:
    batch_tensors = batch_tensors.to(device)  # shape: (64, 12, 8, 8)
    batch_labels = batch_labels.to(device)    # shape: (64,)
    batch_masks = batch_masks.to(device)
    # Forward pass
    logits = network(batch_tensors)        # shape: (64, 1968)
    logits = logits.masked_fill(batch_masks == 0, float('-inf'))   

    # Compute loss
    loss = criterion(logits, batch_labels)

    if torch.isinf(loss):
        optimizer.zero_grad()
        continue

    losses.append(loss.item())
    ut.plot(losses, line_raw, line_smooth, ax)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(network.state_dict(), os.path.join(script_dir, 'training1_weights.pth'))