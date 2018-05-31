from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from torch.autograd import Variable

# resnet deconv block
class DeBottleNeck2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super(DeBottleNeck2d, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        shortcut_padout_0 = stride[0] - 1
        shortcut_padout_1 = stride[1] - 1

        self.pre_act = nn.BatchNorm2d(in_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.ConvTranspose2d(in_channels, out_channels,
                    kernel_size=(1, 1), stride=stride, padding=(0, 0),
                    output_padding=(shortcut_padout_0, shortcut_padout_1))

        padout_10 = stride[0] - kernel_size[0]
        padout_11 = stride[1] - kernel_size[1]
        pad_10 = 0
        pad_11 = 0
        if padout_10 < 0:
            tmp = padout_10 % 2
            pad_10 = (-padout_10 + tmp) // 2
            padout_10 = tmp
        if padout_11 < 0:
            tmp = padout_11 % 2
            pad_11 = (-padout_11 + tmp) // 2
            padout_11 = tmp
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=(pad_10, pad_11), output_padding=(padout_10, padout_11))
        self.bn = nn.BatchNorm2d(out_channels)
        #self.dropout = nn.Dropout2d(p=0.1)
        pad_20 = (kernel_size[0] - 1) // 2
        pad_21 = (kernel_size[1] - 1) // 2
        self.deconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                stride=(1, 1), padding=(pad_20, pad_21))

    def forward(self, x):
        x = F.relu(self.pre_act(x))
        if self.stride != 1 or self.in_channels != self.out_channels:
            y = self.shortcut(x)
        else:
            y = x
        #x = self.dropout(F.relu(self.bn(self.conv1(x))))
        x = F.relu(self.bn(self.deconv1(x)))
        x = self.deconv2(x)
        x = x + y
        return x

# resnet deconv block
class DeBottleNeck1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super(DeBottleNeck1d, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        shortcut_padout_0 = stride - 1

        self.pre_act = nn.BatchNorm1d(in_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.ConvTranspose1d(in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0,
                    output_padding=shortcut_padout_0)

        padout_10 = stride - kernel_size
        pad_10 = 0
        if padout_10 < 0:
            tmp = padout_10 % 2
            pad_10 = (-padout_10 + tmp) // 2
            padout_10 = tmp
        self.deconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=pad_10, output_padding=padout_10)
        self.bn = nn.BatchNorm1d(out_channels)
        self.LRelu = nn.LeakyReLU(0.1)
        pad_20 = (kernel_size - 1) // 2
        self.deconv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                stride=1, padding=pad_20)

    def forward(self, x):
        x = F.relu(self.pre_act(x))
        if self.stride != 1 or self.in_channels != self.out_channels:
            y = self.shortcut(x)
        else:
            y = x
        #x = self.dropout(F.relu(self.bn(self.conv1(x))))
        x = self.LRelu(self.bn(self.deconv1(x)))
        x = self.deconv2(x)
        x = x + y
        return x


class DeBottleNeck1d_3(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, stride, kernel_size, group_num=1):
        super(DeBottleNeck1d_3, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        shortcut_padout_0 = stride - 1

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.ConvTranspose1d(in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0,
                    output_padding=shortcut_padout_0)

        padout_10 = stride - kernel_size
        pad_10 = 0
        if padout_10 < 0:
            tmp = padout_10 % 2
            pad_10 = (-padout_10 + tmp) // 2
            padout_10 = tmp

        #self.deconv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.deconv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.deconv2 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                stride=stride, padding=pad_10, output_padding=padout_10, groups=group_num)

        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.LRelu = nn.LeakyReLU(0.1)

        #self.deconv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.deconv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        if self.stride != 1 or self.in_channels != self.out_channels:
            y = self.shortcut(x)
        else:
            y = x
        x = F.relu(self.bn1(self.deconv1(x)))
        x = self.LRelu(self.bn2(self.deconv2(x)))
        x = self.bn3(self.deconv3(x))
        x = F.relu(x + y)
        return x


class DeBottleNeck1d_3S(nn.Module):
    "using nearest upsampling and conv to avoid artifacts"
    def __init__(self, in_channels, hidden_channels, out_channels, stride, kernel_size, group_num=1):
        super(DeBottleNeck1d_3S, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        shortcut_padout_0 = stride - 1
        pad = (kernel_size-1) // 2

        if stride != 1 or in_channels != out_channels:
            # self.shortcut = nn.ConvTranspose1d(in_channels, out_channels,
            #                                    kernel_size=1, stride=stride, padding=0,
            #                                    output_padding=shortcut_padout_0)
            self.shortcut_upsampling = torch.nn.UpsamplingNearest2d(scale_factor=stride)
            self.shortcut_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.upsampling = torch.nn.UpsamplingNearest2d(scale_factor=stride)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=pad)

        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # self.deconv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        if self.stride != 1 or self.in_channels != self.out_channels:
            y = x.unsqueeze(2)
            y = self.shortcut_upsampling(y)
            y = y[:, :, 0, :]
            y = y.squeeze(2)
            y = self.shortcut_conv(y)
        else:
            y = x
        x = F.relu(self.bn1(self.conv1(x)))

        x = x.unsqueeze(2)
        x = self.upsampling(x)
        x = x[:, :, 0, :]
        x = x.squeeze(2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x + y)
        return x


# group in every layer
# like the block I used in Enc
class DeBottleNeck1d_3G(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, stride, kernel_size, group_num=1):
        super(DeBottleNeck1d_3G, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        shortcut_padout_0 = stride - 1

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.ConvTranspose1d(in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0,
                    output_padding=shortcut_padout_0)

        padout_10 = stride - kernel_size
        pad_10 = 0
        if padout_10 < 0:
            tmp = padout_10 % 2
            pad_10 = (-padout_10 + tmp) // 2
            padout_10 = tmp

        # self.deconv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.deconv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.deconv2 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                stride=stride, padding=pad_10, output_padding=padout_10, groups=group_num)

        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.LRelu = nn.LeakyReLU(0.1)

        #self.deconv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.deconv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        if self.stride != 1 or self.in_channels != self.out_channels:
            y = self.shortcut(x)
        else:
            y = x
        x = F.relu(self.bn1(self.deconv1(x)))
        x = self.LRelu(self.bn2(self.deconv2(x)))
        x = self.bn3(self.deconv3(x))
        x = F.relu(x + y)
        return x


class DeBottleNeck1d_3SE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, stride, kernel_size, group_num=1, r=16):
        super(DeBottleNeck1d_3SE, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        shortcut_padout_0 = stride - 1

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.ConvTranspose1d(in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0,
                    output_padding=shortcut_padout_0)

        padout_10 = stride - kernel_size
        pad_10 = 0
        if padout_10 < 0:
            tmp = padout_10 % 2
            pad_10 = (-padout_10 + tmp) // 2
            padout_10 = tmp

        self.deconv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1,
                                 stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.deconv2 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                stride=stride, padding=pad_10, output_padding=padout_10, groups=group_num)

        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.LRelu = nn.LeakyReLU(0.1)

        self.deconv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1,
                stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.fc1 = nn.Conv1d(out_channels, out_channels // r, kernel_size = 1, stride = 1, padding = 0)
        self.fc2 = nn.Conv1d(out_channels // r, out_channels, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        if self.stride != 1 or self.in_channels != self.out_channels:
            y = self.shortcut(x)
        else:
            y = x
        x = F.relu(self.bn1(self.deconv1(x)))
        x = self.LRelu(self.bn2(self.deconv2(x)))
        x = self.bn3(self.deconv3(x))

        s = F.adaptive_avg_pool1d(x, 1)
        s = F.relu(self.fc1(s))
        s = F.sigmoid(self.fc2(s))
        x = s * x

        x = F.relu(x + y)
        return x