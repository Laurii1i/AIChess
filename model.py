import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mappings import two_way

"""  
      a    b   c   d   e   f   g   h
 1   [ 0,  1,  2,  3,  4,  5,  6,  7]
 2   [ 8,  9, 10, 11, 12, 13, 14, 15]
 3   [16, 17, 18, 19, 20, 21, 22, 23]
 4   [24, 25, 26, 27, 28, 29, 30, 31]
 5   [32, 33, 34, 35, 36, 37, 38, 39]
 6   [40, 41, 42, 43, 44, 45, 46, 47]
 7   [48, 49, 50, 51, 52, 53, 54, 55]
 8   [56, 57, 58, 59, 60, 61, 62, 63]
 
 """
class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers here

        self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1 )
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding = 1 )
        self.LinLayer = nn.Linear(8*8*256, 1968) # Mapping to 1968 move space

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = self.LinLayer(x)
        return x
    
