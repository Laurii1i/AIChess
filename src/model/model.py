
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh


class ResidualBlock(nn.Module):
    """Residual block for ChessNet."""

    def __init__(
        self,
        channels,
        kernel_size=3,
        padding=1,
        norm="batchnorm", # Change this to "groupnorm" for groupnorm normalization
        activation=nn.ReLU,
    ):
        # Conv2d computes weighted sums over local 3×3 regions to extract spatial features from the input planes.
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, bias=False
        )

        # BatchNorm normalizes each feature map as (x - E[x]) / σ_x across the batch.
        self.bn1 = (
            nn.BatchNorm2d(channels)
            if norm == "batchnorm"
            else nn.GroupNorm(8, channels)
        )
        # ReLu maps negative values to zero, keeps positives as is.
        self.act = activation(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = (
            nn.BatchNorm2d(channels)
            if norm == "batchnorm"
            else nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity # ResidualBlock architecture, 𝑥 -> 𝐹(𝑥)+𝑥  --- 𝐹(𝑥) is the out variable, 𝑥 is the identity variable
        out = F.relu(out, inplace=True)
        return out


class ChessNet(nn.Module):
    """Chess neural network with residual blocks, policy head, and value head."""

    def __init__(
        self,
        input_planes=13,
        num_filters=128, # each filter has 128×3×3 = 1152 learnable weights.
        num_res_blocks=10,
        action_size=1968,  # Size of the move space (all chess moves encoded)
        norm="batchnorm",
        dropout=0.0,
    ):
        super().__init__()

        # Stem convolution. This block convert the input into an appropriate shape for the residual block tower

        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters)
            if norm == "batchnorm"
            else nn.GroupNorm(8, num_filters),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters, norm=norm, activation=nn.ReLU)
              for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32) if norm == "batchnorm" else nn.GroupNorm(8, 32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16) if norm == "batchnorm" else nn.GroupNorm(8, 16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming normal initialization: initial filter weights are drawn from a normal distribution 
        with variance = 2 / n, where n is the number of input connections to the neuron.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, 13, 8, 8)

        Returns:
            logits: Tensor of shape (B, action_size)
            v: Tensor of shape (B,) with values in [-1, 1]
        """
        b = x.shape[0]

        # Stem and residual tower
        out = self.stem(x)
        out = self.res_blocks(out) # Pass the input through the residual block tower

        # Policy head
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p, inplace=True)
        p = p.view(b, -1) # Flatten the batch dimension
        logits = self.policy_fc(p) # Final fully connected layer to map feature maps to move-space logits

        # Value head
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v, inplace=True)
        v = v.view(b, -1)
        v = self.value_dropout(v)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = tanh(self.value_fc2(v).squeeze(-1))  # scalar in [-1,1]

        return logits, v
