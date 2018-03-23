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

class BottleNeck1d_3(nn.Module):
    """
    ResNet 3 conv residual block
    batchnorm + preactivation
    dropout used when net is wide
    """

    def __init__(self, in_channels, hidden_channels, out_channels, stride, kernel_size, group_num=1):

        super(BottleNeck1d_3, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1,
                               padding=0)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, groups=group_num)

        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1,
                               padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        if self.stride != 1 or self.in_channels != self.out_channels:
            y = self.shortcut(x)
        else:
            y = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = F.relu(x + y)
        return x

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
        elif self.type == 'cnn':
            width = 16
            self.conv0 = nn.Conv1d(1, width, kernel_size=3, stride=2, padding=4)
            self.block1 = BottleNeck1d_3(in_channels=width, hidden_channels=width//2,
                                         out_channels=width * 2, stride=2, kernel_size=3, group_num=width//4)
            self.block2 = BottleNeck1d_3(in_channels=width * 2, hidden_channels=width//2,
                                         out_channels=width * 2, stride=2, kernel_size=3, group_num=width//4)
            self.block3 = BottleNeck1d_3(in_channels=width * 2, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=3, group_num=width//2)
            self.block4 = BottleNeck1d_3(in_channels=width * 4, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=3, group_num=width//2)
            self.block5 = BottleNeck1d_3(in_channels=width * 4, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=3, group_num=width//2)
            # 16 left
            self.pooling = nn.AvgPool1d(kernel_size=16, stride=16)
            self.fc1 = nn.Linear(width * 4, 1)

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
        elif self.type == 'cnn':
            x = x.unsqueeze(1)
            x = self.conv0(x)
            x = self.block1.forward(x)
            x = self.block2.forward(x)
            x = self.block3.forward(x)
            x = self.block4.forward(x)
            x = self.block5.forward(x)
            x = self.pooling(x)
            x = x.squeeze(2)
            x = self.fc1(x)
            return x

