# chess_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
from pathlib import Path
import chess
import utils as ut
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pyarrow as pa
import pandas as pd
# -------------------------
# Utilities
# -------------------------


def value_target_to_cp(value, clip=600.0):
    """Reverse of cp_to_value_target (approximate)."""
    # inverse tanh (atanh)
    eps = 1e-6
    val = value.clamp(-1+eps, 1-eps)
    return clip * 0.5 * torch.log((1+val)/(1-val))

# -------------------------
# Residual block
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, norm='batchnorm', activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels) if norm == 'batchnorm' else nn.GroupNorm(8, channels)
        self.act = activation(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels) if norm == 'batchnorm' else nn.GroupNorm(8, channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out

# -------------------------
# Main ChessNet
# -------------------------
class ChessNet(nn.Module):
    def __init__(self,
                 input_planes=13,
                 num_filters=128,
                 num_res_blocks=10,
                 action_size=1968,    # set to your move vocabulary size
                 norm='batchnorm',
                 dropout=0.0):
        super().__init__()

        # Stem conv
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters) if norm == 'batchnorm' else nn.GroupNorm(8, num_filters),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters, norm=norm, activation=nn.ReLU) for _ in range(num_res_blocks)]
        )

        # Policy head
        # small conv to reduce channels then fully connected to action_size
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32) if norm == 'batchnorm' else nn.GroupNorm(8, 32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16) if norm == 'batchnorm' else nn.GroupNorm(8, 16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # initialization (kaiming)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 13, 8, 8)
        
        b = x.shape[0]
        out = self.stem(x)
        out = self.res_blocks(out)

        # policy
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p, inplace=True)
        p = p.view(b, -1)
        logits = self.policy_fc(p)   # (B, action_size)

        # value
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v, inplace=True)
        v = v.view(b, -1)
        v = self.value_dropout(v)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = torch.tanh(self.value_fc2(v).squeeze(-1))  # scalar in [-1,1]

        return logits, v

# -------------------------
# Loss wrapper and training step example
# -------------------------
class Trainer:
    def __init__(self, model, action_weight=1.0, value_weight=1.0, lr=3e-4, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.action_weight = action_weight
        self.value_weight = value_weight

    def compute_loss(self, logits, pred_value, policy_target, value_target):
        # policy_target: long (B,) indices; logits: (B,action_size)
        # cp_target: raw centipawns (B,) float
        # convert cp to normalized target in [-1,1]

        # policy loss: cross-entropy (PyTorch expects raw logits)
        policy_loss = F.cross_entropy(logits, policy_target.to(self.device))

        valid_mask = ~torch.isnan(value_target)
        if valid_mask.sum() > 0:
            value_loss = F.smooth_l1_loss(
                pred_value[valid_mask], value_target[valid_mask]
            )
        else:
            # If all are NaN, set value loss to 0 (avoid NaN total loss)
            value_loss = torch.tensor(0.0, device=self.device)

        total_loss = self.action_weight * policy_loss + self.value_weight * value_loss
        return total_loss, policy_loss.detach(), value_loss.detach()

    def train_epoch(self, dataloader, epoch=0, max_grad_norm=None, plot_package = None, losses = []):
        self.model.train()
        for i, (board, policy_idx, mask, value_target) in enumerate(dataloader):
            board = board.to(self.device)
            policy_idx = policy_idx.to(self.device)
            value_target = value_target.to(self.device)
            mask = mask.to(self.device)

            logits, v = self.model(board)
            logits = logits.masked_fill(mask == 0, float('-inf')) 
            loss, ploss, vloss = self.compute_loss(logits, v, policy_idx, value_target)

            self.optim.zero_grad()
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optim.step()

            losses.append(loss.item())
            if plot_package is not None:
                ax, line_raw, line_smooth = plot_package
                ut.plot(losses, line_raw, line_smooth, ax)

# -------------------------
# Example usage (skeleton)
# -------------------------
if __name__ == "__main__":

    batch_size = 128
    board = chess.Board()

    script_dir = Path(__file__).resolve().parent

    plt.ion()
    fig, ax = plt.subplots()
    line_raw, = ax.plot([], [], 'b-', alpha=0.3, label='Batch Loss')    # raw batch loss
    line_smooth, = ax.plot([], [], 'r-', label='Smoothed Loss')         # moving average
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Evolution")
    ax.legend()

    action_size = 1968
    model = ChessNet(input_planes=13, num_filters=128, num_res_blocks=10,
                    action_size=action_size, norm='batchnorm', dropout=0.1)
    trainer = Trainer(model, action_weight=1.0, value_weight=1.0, lr=2e-4)
    plot_package = [ax, line_raw, line_smooth]

    losses_track = []
    model.load_state_dict(torch.load(r"C:\Users\lauri\Documents\Chess\AIChess\improved_model1_groups_0-14.pth"))
    model.train()
    for i in range(2, 7):

        file_path = script_dir / Path(f'../TrainingData/train-{i:05d}-of-00014.parquet_edited')
        print(f'{i}/{6} starting')
        parquet_file = pq.ParquetFile(file_path)
        #df = pd.read_parquet(file_path, columns= ['fen', 'cp', 'line'])

        #df = df.drop_duplicates(subset=['fen'], keep='first')

        #table = pa.Table.from_pandas(df)

        # Write to Parquet file with specified row group size
        #pq.write_table(table, script_dir / Path(f'../TrainingData/train-{i:05d}-of-00014.parquet_edited'), row_group_size=100_000)

        for k in range(2, parquet_file.num_row_groups):
            table = parquet_file.read_row_group(k, columns=['fen', 'line', 'cp'])
            df = table.to_pandas()
            #df = pd.read_parquet(file_path, columns = ['fen', 'line', 'cp'])
            #df = df.drop_duplicates(subset=['fen'])

            #df = df.iloc[:200000].reset_index(drop=True)
            df['value_target'] = df['cp'].apply(ut.cp_to_value_target)
            df['best_move'] = df.apply(ut.line_to_best_move, axis = 1)
            df = df[df['best_move'] != 'nan'].reset_index(drop=True)
            df['tensor'] = df['fen'].apply(ut.fen_to_tensor) # 48 seconds for million rows
            df['best_move_labels'] = df['best_move'].map(uci_index_connector)
            df['legal_moves_mask'] = df['fen'].apply(ut.get_mask)
            
            # Save to Parquet with row groups of 100,000 rows
            '''pq.write_table(
                table,
                'train_subset.parquet',
                row_group_size=100_000,  # 100k rows per internal row group
                compression='snappy',    # optional, recommended
            )'''


            all_tensors = torch.stack(df['tensor'].tolist())                 # shape: (num_rows, 12, 8, 8)
            all_labels = torch.tensor(df['best_move_labels'].tolist(), dtype=torch.long)
            all_masks  = torch.stack(df['legal_moves_mask'].tolist())        # shape: (num_rows, 1968)
            all_value_targets = torch.tensor(df['value_target'].to_list())
            

            dataset = TensorDataset(all_tensors, all_labels, all_masks, all_value_targets)


            # Example DataLoader (user should replace with their dataset)

            train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
            trainer.train_epoch(train_loader, epoch=0, plot_package = plot_package, losses = losses_track)

        torch.save(model.state_dict(), "improved_model1_groups_0-14.pth")        

