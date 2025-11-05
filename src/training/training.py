# trainer.py

# -----------------------------
# Standard library imports
# -----------------------------
from pathlib import Path

# -----------------------------
# Third-party imports
# -----------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Local application imports
# -----------------------------
from config import ROOT
from mappings import uci_index_connector
import utils as ut

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        action_weight: float = 1.0,
        value_weight: float = 1.0,
        lr: float = 3e-4,
        device: str = 'cuda',
        batch_size: int = 64,
    ):
        """
        Initialize the Trainer for a chess neural network.

        Args:
            model (nn.Module): PyTorch neural network model.
            action_weight (float, optional): Weight for the policy loss. Defaults to 1.0.
            value_weight (float, optional): Weight for the value loss. Defaults to 1.0.
            lr (float, optional): Learning rate for the optimizer. Defaults to 3e-4.
            device (str, optional): Device to train on ('cuda' or 'cpu'). Defaults to 'cuda'.
            batch_size (int, optional): Batch size for training. Defaults to 128.
        """

        # Determine device: GPU if available, otherwise CPU
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Initialize and move model to device
        self.model = model.to(self.device)

        # Load pretrained model weights
        self.model_weights_path: Path = ROOT / "src" / "model" / "model_weights.pth"
        self.model.load_state_dict(torch.load(self.model_weights_path))
        self.model.train()  # Set model to training mode

        # Initialize optimizer with weight decay
        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Store loss weights
        self.action_weight = action_weight
        self.value_weight = value_weight

        # Load training data Parquet file
        self.training_data_file = pq.ParquetFile(
            ROOT / "src" / "training" / "data" / "example_training_data"
        )

        # Store batch size
        self.batch_size = batch_size

        # Initialize real-time loss plotting
        self.initiate_loss_plot()


    def initiate_loss_plot(self):
        """
        Initialize an interactive matplotlib plot for tracking training loss.

        Creates two lines:
            - Blue (semi-transparent): raw batch loss.
            - Red: smoothed moving average of the loss.

        The plot is interactive and intended for live updates during training.
        """

        # Enable interactive plotting (non-blocking updates)
        plt.ion()

        # Initialize loss storage
        self.losses = []

        # Create figure and axis for the loss plot
        self.fig, self.ax = plt.subplots()

        # Initialize line for raw (per-batch) loss values
        self.line_raw, = self.ax.plot(
            [], [], "b-", alpha=0.3, label="Batch Loss"
        )

        # Initialize line for smoothed (moving-average) loss values
        self.line_smooth, = self.ax.plot(
            [], [], "r-", label="Smoothed Loss"
        )

        # Label the axes and set title
        self.ax.set_xlabel("Batch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss Evolution")

        # Add legend for clarity
        self.ax.legend()

    def compute_loss(self, logits, pred_value, policy_target, value_target):
        """
        Compute the combined loss for policy and value predictions.

        Args:
            logits (Tensor): Raw policy logits of shape (B, action_size).
            pred_value (Tensor): Predicted value head output of shape (B,).
            policy_target (Tensor): Target move indices (LongTensor of shape (B,)).
            value_target (Tensor): Target position values (FloatTensor of shape (B,)).

        Returns:
            tuple(Tensor, Tensor, Tensor):
                - total_loss: Weighted sum of policy and value losses.
                - policy_loss: Cross-entropy loss (detached).
                - value_loss: Smooth L1 loss (detached).
        """

        # Policy loss: standard cross-entropy (logits are unnormalized)
        policy_loss = F.cross_entropy(logits, policy_target.to(self.device))

        # Some positions may not have a valid value target (e.g., from missing evals).
        # Compute value loss only for valid targets (non-NaN entries).
        valid_mask = ~torch.isnan(value_target)
        if valid_mask.any():
            value_loss = F.smooth_l1_loss(
                pred_value[valid_mask], value_target[valid_mask]
            )
        else:
            # If all targets are NaN, avoid propagating NaN by assigning zero loss.
            value_loss = torch.tensor(0.0, device=self.device)

        # Combine policy and value losses using user-defined weights
        total_loss = (
            self.action_weight * policy_loss +
            self.value_weight * value_loss
        )

        # Return total and individual losses (detached for logging)
        return total_loss, policy_loss.detach(), value_loss.detach()


    def train(self):
        """
        Train the model using data stored in a Parquet file.

        The Parquet file is read one row group at a time to avoid memory issues.
        For each row group:
            1. Convert it to a Pandas DataFrame.
            2. Preprocess it (encode FENs, moves, masks, values).
            3. Create a DataLoader.
            4. Train the model on the loaded data.
        """

        num_groups = self.training_data_file.num_row_groups
        # Loop over all row groups in the Parquet file
        for k in range(num_groups):

            print(f'Starting to process training data group {k} / {num_groups - 1}')
            # Read only the necessary columns for training
            table = self.training_data_file.read_row_group(
                k, columns=['fen', 'line', 'cp']
            )

            # Convert PyArrow Table to Pandas DataFrame
            df: pd.DataFrame = table.to_pandas()

            # Preprocess DataFrame: compute value targets, tensors, move masks, etc.
            df = self.modify_dataframe(df)

            # Create a PyTorch DataLoader from the preprocessed DataFrame
            dataloader: DataLoader = self.create_data_loader(df)

            # Train the model on this batch of data
            self.train_loaded_data(dataloader)

        # Save the updated weights
        torch.save(self.model.state_dict(), self.model_weights_path) 


    def create_data_loader(self, df: pd.DataFrame) -> DataLoader:
        """
        Create a PyTorch DataLoader from a preprocessed DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing encoded tensors, labels,
                masks, and value targets (as produced by `modify_dataframe`).

        Returns:
            DataLoader: A DataLoader that yields batches of
                (board_tensor, best_move_label, legal_moves_mask, value_target).
        """

        # Stack board tensors into a single tensor.
        # Shape: (num_positions, 12, 8, 8)
        all_tensors = torch.stack(df["tensor"].tolist())

        # Convert best move labels (indices 0–1967) to a LongTensor for classification.
        all_labels = torch.tensor(df["best_move_labels"].tolist(), dtype=torch.long)

        # Stack legal move masks (bool or float tensors) into a single tensor.
        # Shape: (num_positions, 1968)
        all_masks = torch.stack(df["legal_moves_mask"].tolist())

        # Convert normalized centipawn-based value targets to a float tensor.
        # Shape: (num_positions,)
        all_value_targets = torch.tensor(df["value_target"].tolist(), dtype=torch.float32)

        # Combine all tensors into a single dataset for easy batching.
        dataset = TensorDataset(all_tensors, all_labels, all_masks, all_value_targets)

        # Create the DataLoader.
        # - batch_size: number of samples per batch
        # - shuffle: randomize sample order each epoch
        # - num_workers: parallel data loading processes
        # - pin_memory: speeds up transfer to GPU if available
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        ) 

    def modify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add essential training columns to the given DataFrame.

        This function augments the DataFrame with value targets, encoded tensors,
        move labels, and legal move masks derived from chess position data.
        """

        # Convert centipawn evaluation (cp) into a normalized target value
        # scaled between -1 (black winning) and 1 (white winning).
        df["value_target"] = df["cp"].apply(ut.cp_to_value_target)

        # Extract the best move from the 'line' field.
        # Each line represents an optimal sequence of moves; we take the first one.
        df["best_move"] = df.apply(ut.line_to_best_move, axis=1)
        df = df[(df['best_move'] != 'nan')]

        # Filter out rows where the best move could not be determined.
        # Use .notna() instead of comparing to the string "nan".
        df = df[df["best_move"].notna()].reset_index(drop=True)

        # Encode FEN strings (board positions) into tensor representations
        # suitable as model inputs.
        df["tensor"] = df["fen"].apply(ut.fen_to_tensor)

        # Convert each best move from UCI notation into its corresponding
        # integer label in the move space (0–1967).
        df["best_move_labels"] = df["best_move"].map(uci_index_connector)

        # Create a boolean mask of length 1968, where True marks legal moves.
        # This mask will be used to disable invalid policy outputs.
        df["legal_moves_mask"] = df["fen"].apply(lambda fen: ut.get_mask(fen, uci_index_connector))


        return df


    def train_loaded_data(self, dataloader, max_grad_norm=None):
        """
        Train the model for one epoch.

        Args:
            dataloader (DataLoader): Iterable providing training batches.
            epoch (int, optional): Current epoch index. Defaults to 0.
            max_grad_norm (float, optional): Maximum allowed gradient norm for clipping.
        """
        for board, policy_idx, mask, value_target in dataloader:

            # Move batch tensors to the correct device (CPU or GPU)
            board, policy_idx, mask, value_target = self.move_batch_to_device(
                board, policy_idx, mask, value_target
            )

            # Forward pass: compute policy logits and value predictions
            logits, v = self.model(board)

            # Mask out invalid moves by setting their logits to negative infinity
            logits = logits.masked_fill(mask == 0, float('-inf'))

            # Compute total loss (and optionally, policy/value losses separately)
            loss, _, _ = self.compute_loss(logits, v, policy_idx, value_target)

            # Clear gradients from the previous iteration
            self.optim.zero_grad()

            # Backward pass: compute gradients of the loss w.r.t. model parameters
            loss.backward()

            if max_grad_norm:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            # Update model parameters using the optimizer
            self.optim.step()

            # Track loss for monitoring or visualization
            self.losses.append(loss.item())

            # Optionally update plots or logs (can be throttled for performance)
            self.plot_loss()


    def move_batch_to_device(self, *args):
        """
        Move one or more tensors to the device specified by self.device.

        Args:
            *args: One or more torch.Tensor objects.

        Returns:
            List[torch.Tensor]: A list of tensors moved to self.device.
        """
        return [arg.to(self.device) for arg in args]

    def plot_loss(self, window=10):
        """
        Update the loss plot dynamically.
        - losses: list of batch losses
        - line_raw, line_smooth: Line2D objects from ax.plot()
        - ax: matplotlib axes
        - window: moving average window
        """
        if len(self.losses) == 0:
            return

        x = np.arange(len(self.losses))
        
        if len(self.losses) < window:
            smoothed = np.array(self.losses)
        else:
            smoothed = np.convolve(self.losses, np.ones(window) / window, mode='same')

        self.line_raw.set_data(x, self.losses)
        self.line_smooth.set_data(x, smoothed)

        self.ax.relim()
        self.ax.autoscale_view()

        plt.pause(0.001)


           

