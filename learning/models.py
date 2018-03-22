from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

from learning.npn import NPNLinear
from learning.npn import NPNRelu
from learning.npn import NPNSigmoid
from learning.npn import NPNDropout
from learning.npn import KL_loss
from learning.npn import L2_loss


class Enc(nn.Module):
    def __init__(self, args):
        super(Enc, self).__init__()
        self.type = args.enc_type
        if self.type == 'mlp':
            width = 64
            self.fc1 = nn.Linear(INPUT_DIM, width)
            self.fc2 = nn.Linear(width, width)
            self.fc3 = nn.Linear(width, 1)

            # self.dropout1 = nn.Dropout(p=0.2)
            # self.dropout2 = nn.Dropout(p=0.2)
            self.bn1 = nn.BatchNorm1d(width)
            self.bn2 = nn.BatchNorm1d(width)
        elif self.type == 'npn':
            width = 128
            self.fc1 = NPNLinear(INPUT_DIM, width, dual_input=False)
            self.nonlinear1 = NPNRelu()
            # self.dropout1 = NPNDropout(self.fc_drop)
            self.fc2 = NPNLinear(width, 1)
            # self.nonlinear2 = NPNSigmoid()

    def forward(self, x):
        if self.type == 'mlp':
            x = F.relu(self.bn1(self.fc1(x)))
            # x = self.dropout1(x)
            x = F.relu(self.bn2(self.fc2(x)))
            # x = self.dropout2(x)
            x = self.fc3(x)
            return x

        elif self.type == 'npn':
            x = self.nonlinear1(self.fc1(x))
            # x = self.dropout1(x)
            x = self.fc2(x)
            # x, s = self.nonlinear2(x)
            a_m, a_s = x

            return a_m, a_s
