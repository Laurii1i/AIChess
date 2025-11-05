
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh

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
        v = tanh(self.value_fc2(v).squeeze(-1))  # scalar in [-1,1]

        return logits, v