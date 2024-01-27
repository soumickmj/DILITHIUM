#!/usr/bin/env python

# from __future__ import print_function, division
"""

Purpose :

"""
import torch.nn
import torch
import torch.nn as nn

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class ConvBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.PReLU(num_parameters=out_channels, init=0.25),
            # nn.Dropout3d(),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.PReLU(num_parameters=out_channels, init=0.25),
            # nn.Dropout3d(),
            nn.BatchNorm3d(num_features=out_channels))

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparableConvBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, bias=True):
        super(SeparableConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                      bias=bias),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.PReLU(num_parameters=out_channels, init=0.25),
            # nn.Dropout3d(),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1,
                      bias=bias),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding, bias=bias),
            nn.PReLU(num_parameters=out_channels, init=0.25),
            # nn.Dropout3d(),
            nn.BatchNorm3d(num_features=out_channels))

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    """
    Up Convolution Block
    """

    # def __init__(self, in_ch, out_ch):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=padding),
            nn.BatchNorm3d(num_features=out_channels),
            nn.PReLU(num_parameters=out_channels, init=0.25))

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """
    Attention Block
    """

    def __init__(self, f_g, f_l, f_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(f_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttUnet(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=1, output_ch=6):
        super(AttUnet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, filters[0])
        self.Conv2 = SeparableConvBlock(filters[0], filters[1])
        self.Conv3 = SeparableConvBlock(filters[1], filters[2])
        self.Conv4 = SeparableConvBlock(filters[2], filters[3])
        self.Conv5 = SeparableConvBlock(filters[3], filters[4])

        self.Up5 = UpConv(filters[4], filters[3])
        self.Att5 = AttentionBlock(f_g=filters[3], f_l=filters[3], f_int=filters[2])
        self.Up_conv5 = SeparableConvBlock(filters[4], filters[3])

        self.Up4 = UpConv(filters[3], filters[2])
        self.Att4 = AttentionBlock(f_g=filters[2], f_l=filters[2], f_int=filters[1])
        self.Up_conv4 = SeparableConvBlock(filters[3], filters[2])

        self.Up3 = UpConv(filters[2], filters[1])
        self.Att3 = AttentionBlock(f_g=filters[1], f_l=filters[1], f_int=filters[0])
        self.Up_conv3 = SeparableConvBlock(filters[2], filters[1])

        self.Up2 = UpConv(filters[1], filters[0])
        self.Att2 = AttentionBlock(f_g=filters[0], f_l=filters[0], f_int=32)
        self.Up_conv2 = ConvBlock(filters[1], filters[0])

        self.Conv = nn.Conv3d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        x4 = self.Att5(d5, e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(d4, e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(d3, e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(d2, e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out
