from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Enc(nn.Module):
    def __init__(self):
        super(Enc, self).__init__()
        width = 64
        self.fc1 = nn.Linear(INPUT_DIM, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1)

        # self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x
