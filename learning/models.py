from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from torch.autograd import Variable

from learning.npn import NPNLinear
from learning.npn import NPNRelu
from learning.npn import NPNSigmoid
from learning.npn import NPNDropout
from learning.npn import KL_loss
from learning.npn import L2_loss
from learning.deconv_block import *

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
        self.fc_drop = 0.5
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
            self.fc1 = NPNLinear(INPUT_DIM, width, dual_input=False, first_layer_assign=True)
            self.nonlinear1 = NPNRelu()
            # self.dropout1 = NPNDropout(self.fc_drop)
            self.fc2 = NPNLinear(width, 1)
            # self.nonlinear2 = NPNSigmoid()
        elif self.type == 'cnn':
            width = 32
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
            self.pooling = nn.AvgPool1d(kernel_size=4, stride=4)
            self.fc1 = nn.Linear(width * 4 * 4, width * 4 * 4)
            self.fc2 = nn.Linear(width * 4 * 4, 1)
            self.dropout = nn.Dropout(0.5)
        elif self.type == 'cnn1':
            width = args.cnn_width
            self.conv0 = nn.Conv1d(1, width, kernel_size=3, stride=2, padding=4)
            self.block0 = BottleNeck1d_3(in_channels=width, hidden_channels=width // 4,
                                         out_channels=width, stride=2, kernel_size=3, group_num=width // 8)
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
            self.pooling = nn.AvgPool1d(kernel_size=8, stride=8)
            self.fc1 = nn.Linear(width * 4, 1)
        elif self.type == 'combined':
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
            self.fc1 = NPNLinear(width * 4, width * 4, dual_input=False, first_layer_assign=False)
            self.nonlinear1 = NPNRelu()
            # self.dropout1 = NPNDropout(self.fc_drop)
            self.fc2 = NPNLinear(width * 4, 1)
        elif self.type == 'combined_dis':
            width = 16
            kernel_size = 3
            self.conv0 = nn.Conv1d(1, width, kernel_size=3, stride=2, padding=4)
            self.block1 = BottleNeck1d_3(in_channels=width, hidden_channels=width//2,
                                         out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width//4)
            self.block2 = BottleNeck1d_3(in_channels=width * 2, hidden_channels=width//2,
                                         out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width//4)
            self.block3 = BottleNeck1d_3(in_channels=width * 2, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width//2)
            self.block4 = BottleNeck1d_3(in_channels=width * 4, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width//2)
            self.block5 = BottleNeck1d_3(in_channels=width * 4, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width//2)
            # 16 left
            self.pooling = nn.AvgPool1d(kernel_size=16, stride=16)
            self.fc1 = NPNLinear(width * 4 + 1, width * 4, dual_input=False, first_layer_assign=False)
            self.nonlinear1 = NPNRelu()
            # self.dropout1 = NPNDropout(self.fc_drop)
            self.fc2 = NPNLinear(width * 4, 1)

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
            x_size = x.size()
            # x = x.squeeze(2)
            x = x.view(x_size[0], x_size[1] * x_size[2])
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
        elif self.type == 'cnn1':
            x = x.unsqueeze(1)
            x = self.conv0(x)
            x = self.block0.forward(x)
            x = self.block1.forward(x)
            x = self.block2.forward(x)
            x = self.block3.forward(x)
            x = self.block4.forward(x)
            x = self.block5.forward(x)
            x = self.pooling(x)
            x = x.squeeze(2)
            x = self.fc1(x)
            return x
        elif self.type == 'combined':
            x = x.unsqueeze(1)
            x = self.conv0(x)
            x = self.block1.forward(x)
            x = self.block2.forward(x)
            x = self.block3.forward(x)
            x = self.block4.forward(x)
            x = self.block5.forward(x)
            x = self.pooling(x)
            x = x.squeeze(2)
            x = self.nonlinear1(self.fc1(x))
            # x = self.dropout1(x)
            x = self.fc2(x)
            a_m, a_s = x
            return a_m, a_s
        elif self.type == 'combined_dis':
            wave, dis = x
            x = wave.unsqueeze(1)
            dis = dis.unsqueeze(1)
            x = self.conv0(x)
            x = self.block1.forward(x)
            x = self.block2.forward(x)
            x = self.block3.forward(x)
            x = self.block4.forward(x)
            x = self.block5.forward(x)
            x = self.pooling(x)
            # x = x.squeeze(2)
            x_size = x.size()
            x = x.view(x_size[0], x_size[1] * x_size[2])
            x = torch.cat((dis, x), dim=1)
            x = self.nonlinear1(self.fc1(x))
           # x = self.dropout1(x)
            x = self.fc2(x)
            a_m, a_s = x
            return a_m, a_s


class VaeEnc(nn.Module):
    def __init__(self, args):
        super(VaeEnc, self).__init__()
        self.type = args.enc_type
        if self.type == 'vae':
            self.width = 64
            width = self.width
            kernel_size = 5
            self.conv0 = nn.Conv1d(1, width, kernel_size=kernel_size, stride=2, padding=4)
            self.block1 = BottleNeck1d_3(in_channels=width, hidden_channels=width // 2,
                                         out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.block2 = BottleNeck1d_3(in_channels=width * 2, hidden_channels=width // 2,
                                         out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.block3 = BottleNeck1d_3(in_channels=width * 2, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width // 2)
            self.block4 = BottleNeck1d_3(in_channels=width * 4, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width // 2)
            self.block5 = BottleNeck1d_3(in_channels=width * 4, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width // 2)
            # 16 left
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

            self.fc1 = NPNLinear(width * 2 * 8 + 1, width * 2)
            self.nonlinear1 = NPNRelu()
            # self.dropout1 = NPNDropout(self.fc_drop)
            self.fc2 = NPNLinear(width * 2, 1)

    def forward(self, x):
        if self.type == 'vae':
            wave, dis = x
            x = wave.unsqueeze(1)
            dis = dis.unsqueeze(1)
            x = self.conv0(x)
            x = self.block1.forward(x)
            x = self.block2.forward(x)
            x = self.block3.forward(x)
            x = self.block4.forward(x)
            x = self.block5.forward(x)
            x = self.pooling(x)

            mean = x[:, :self.width * 2, :]
            mean = mean.contiguous()
            stddev = F.softplus(x[:, self.width * 2:, :])
            stddev = stddev.contiguous()
            mean = mean.view(mean.size(0), mean.size(1) * mean.size(2))
            stddev = stddev.view(stddev.size(0), stddev.size(1) * stddev.size(2))

            normal_array = Variable(torch.normal(means=torch.zeros(mean.size()), std=1.0).cuda())
            z = mean + stddev * normal_array

            # x = torch.cat((dis, z), dim=1)  # this is one solution
            x_m = torch.cat((dis, mean), dim=1)
            x_s = torch.cat((Variable(torch.zeros((x_m.size(0), 1)).cuda()), stddev), dim=1)
            x = x_m, x_s

            x = self.nonlinear1(self.fc1(x))
            # x = self.dropout1(x)
            x = self.fc2(x)
            a_m, a_s = x
            return a_m, a_s, mean, stddev, z


class AEEnc(nn.Module):
    def __init__(self, args):
        super(AEEnc, self).__init__()
        self.type = args.enc_type
        if self.type == 'AE':
            self.width = 64
            width = self.width
            kernel_size = 5
            self.conv0 = nn.Conv1d(1, width, kernel_size=kernel_size, stride=2, padding=4)
            self.block1 = BottleNeck1d_3(in_channels=width, hidden_channels=width // 2,
                                         out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.block2 = BottleNeck1d_3(in_channels=width * 2, hidden_channels=width // 2,
                                         out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.block3 = BottleNeck1d_3(in_channels=width * 2, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width // 2)
            self.block4 = BottleNeck1d_3(in_channels=width * 4, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width // 2)
            self.block5 = BottleNeck1d_3(in_channels=width * 4, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width // 2)
            # 16 left
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

            self.fc1 = NPNLinear(width * 4 * 8 + 1, width * 2, dual_input=False)
            self.nonlinear1 = NPNRelu()
            # self.dropout1 = NPNDropout(self.fc_drop)
            self.fc2 = NPNLinear(width * 2, 1)

    def forward(self, x):
        if self.type == 'AE':
            wave, dis = x
            x = wave.unsqueeze(1)
            dis = dis.unsqueeze(1)
            x = self.conv0(x)
            x = self.block1.forward(x)
            x = self.block2.forward(x)
            x = self.block3.forward(x)
            x = self.block4.forward(x)
            x = self.block5.forward(x)
            z = self.pooling(x)

            zz = z.view(z.size(0), z.size(1) * z.size(2))
            zz = torch.cat((dis, zz), dim=1)
            x = self.nonlinear1(self.fc1(zz))
            # x = self.dropout1(x)
            x = self.fc2(x)
            a_m, a_s = x
            return a_m, a_s, z


class VaeDec(nn.Module):
    def __init__(self, args):
        super(VaeDec, self).__init__()
        self.type = args.enc_type
        if self.type == 'vae' or self.type == 'vaemlp':
            width = 32
            kernel_size = 5
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
            self.de_block1 = DeBottleNeck1d_3G(in_channels=width * 4, hidden_channels=width,
                                               out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width // 2)
            # self.de_block1_1 = DeBottleNeck1d_3G(in_channels=width * 4, hidden_channels=width,
            #                                      out_channels=width * 4, stride=2, kernel_size=3, group_num=width // 2)
            # self.de_block1_2 = DeBottleNeck1d_3G(in_channels=width * 4, hidden_channels=width,
            #                                      out_channels=width * 4, stride=2, kernel_size=3, group_num=width // 2)
            # self.de_block1_3 = DeBottleNeck1d_3G(in_channels=width * 4, hidden_channels=width,
            #                                      out_channels=width * 4, stride=2, kernel_size=3, group_num=width // 2)
            # self.de_block1_4 = DeBottleNeck1d_3G(in_channels=width * 4, hidden_channels=width,
            #                                      out_channels=width * 4, stride=2, kernel_size=3, group_num=width // 2)
            self.de_block2 = DeBottleNeck1d_3G(in_channels=width * 4, hidden_channels=width,
                                               out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 2)
            self.de_block3 = DeBottleNeck1d_3G(in_channels=width * 2, hidden_channels=width // 2,
                                               out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.de_block4 = DeBottleNeck1d_3G(in_channels=width * 2, hidden_channels=width // 2,
                                               out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.de_block5 = DeBottleNeck1d_3G(in_channels=width * 2, hidden_channels=width // 2,
                                               out_channels=width, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.deconv = nn.ConvTranspose1d(width, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        if self.type == 'vae' or self.type == 'vaemlp':
            x = x.view(x.size(0), x.size(1) // 8, 8)
            x = self.upsample_layer(x)
            # x = torch.cat((x,x,x,x,x,x, x,x,x,x,x,x, x,x,x,x), dim=2)
            # x = torch.cat((x, x, x, x), dim=2)
            x = self.de_block1.forward(x)
            # x = self.de_block1_1.forward(x)
            # x = self.de_block1_2.forward(x)
            # x = self.de_block1_3.forward(x)
            # x = self.de_block1_4.forward(x)
            x = self.de_block2.forward(x)
            x = self.de_block3.forward(x) #96
            x = self.de_block4.forward(x) # 192
            x = self.de_block5.forward(x) # 384
            x = self.deconv(x)
            x = x.squeeze(1)
            x = x[:, 4:-4]
            return x


class AEDec(nn.Module):
    def __init__(self, args):
        super(AEDec, self).__init__()
        self.type = args.enc_type
        if self.type == 'AE':
            width = 64
            kernel_size = 5
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
            self.de_block1 = DeBottleNeck1d_3G(in_channels=width * 4, hidden_channels=width,
                                               out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width // 2)
            self.de_block2 = DeBottleNeck1d_3G(in_channels=width * 4, hidden_channels=width,
                                               out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 2)
            self.de_block3 = DeBottleNeck1d_3G(in_channels=width * 2, hidden_channels=width // 2,
                                               out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.de_block4 = DeBottleNeck1d_3G(in_channels=width * 2, hidden_channels=width // 2,
                                               out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.de_block5 = DeBottleNeck1d_3G(in_channels=width * 2, hidden_channels=width // 2,
                                               out_channels=width, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.deconv = nn.ConvTranspose1d(width, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        if self.type == 'AE':
            x = self.upsample_layer(x)
            x = self.de_block1.forward(x)
            x = self.de_block2.forward(x)
            x = self.de_block3.forward(x)  #96
            x = self.de_block4.forward(x) # 192
            x = self.de_block5.forward(x) # 384
            x = self.deconv(x)
            x = x.squeeze(1)
            x = x[:, 4:-4]
            return x


class VaeMlpEnc(nn.Module):
    def __init__(self, args):
        super(VaeMlpEnc, self).__init__()
        self.type = args.enc_type
        if self.type == 'vaemlp':
            self.width = 64
            width = self.width
            kernel_size = 5
            self.conv0 = nn.Conv1d(1, width, kernel_size=kernel_size, stride=2, padding=4)
            self.block1 = BottleNeck1d_3(in_channels=width, hidden_channels=width // 2,
                                         out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.block2 = BottleNeck1d_3(in_channels=width * 2, hidden_channels=width // 2,
                                         out_channels=width * 2, stride=2, kernel_size=kernel_size, group_num=width // 4)
            self.block3 = BottleNeck1d_3(in_channels=width * 2, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width // 2)
            self.block4 = BottleNeck1d_3(in_channels=width * 4, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width // 2)
            self.block5 = BottleNeck1d_3(in_channels=width * 4, hidden_channels=width,
                                         out_channels=width * 4, stride=2, kernel_size=kernel_size, group_num=width // 2)
            # 16 left
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)

            self.fc1 = nn.Linear(width * 2 * 8 + 1, width * 2)
            self.fc2 = nn.Linear(width * 2, 1)

    def forward(self, x):
        if self.type == 'vaemlp':
            wave, dis = x
            x = wave.unsqueeze(1)
            dis = dis.unsqueeze(1)
            x = self.conv0(x)
            x = self.block1.forward(x)
            x = self.block2.forward(x)
            x = self.block3.forward(x)
            x = self.block4.forward(x)
            x = self.block5.forward(x)
            x = self.pooling(x)

            mean = x[:, :self.width * 2, :]
            mean = mean.contiguous()
            stddev = F.softplus(x[:, self.width * 2:, :])
            stddev = stddev.contiguous()
            mean = mean.view(mean.size(0), mean.size(1) * mean.size(2))
            stddev = stddev.view(stddev.size(0), stddev.size(1) * stddev.size(2))

            normal_array = Variable(torch.normal(means=torch.zeros(mean.size()), std=1.0).cuda())
            z = mean + stddev * normal_array

            # x = torch.cat((dis, z), dim=1)  # this is one solution
            x_m = torch.cat((dis, mean), dim=1)
            # x_s = torch.cat((Variable(torch.zeros((x_m.size(0), 1)).cuda()), stddev), dim=1)
            # x = x_m, x_s

            x = F.relu(self.fc1(x_m))
            # x = self.dropout1(x)
            x = self.fc2(x)
            return x, mean, stddev, z
