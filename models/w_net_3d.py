#!/usr/bin/env python

# from __future__ import print_function, division
"""

Purpose :

"""
import torch.nn
import torch
import torch.nn as nn
from models.attention_unet3d import AttUnet
from models.unet3d import UNet, UNetDeepSup

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class WNet3dAttUNet(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=1, output_ch=6):
        super(WNet3dAttUNet, self).__init__()

        self.Encoder = AttUnet(img_ch=img_ch, output_ch=output_ch)
        self.Decoder = AttUnet(img_ch=output_ch, output_ch=1)

        self.activation = torch.nn.Softmax(dim=1)

        self.Conv = nn.Conv3d(output_ch, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, ip, ip_mask=None, ops="both"):
        encoder_op = self.Encoder(ip)
        if ip_mask is not None:
            encoder_op = ip_mask * encoder_op
        class_prob = self.activation(encoder_op)
        feature_rep = self.Conv(encoder_op)
        if ops == "enc":
            return class_prob, feature_rep
        reconstructed_op = self.Decoder(class_prob)
        # if ip_mask is not None:
        #     reconstructed_op = torch.amax(ip_mask, dim=1, keepdim=True) * reconstructed_op
        if ops == "dec":
            return reconstructed_op
        if ops == "both":
            return class_prob, feature_rep, reconstructed_op
        else:
            raise ValueError('Invalid ops, ops must be in [enc, dec, both]')


class WNet3dUNet(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=1, output_ch=6):
        super(WNet3dUNet, self).__init__()

        self.Encoder = UNet(in_ch=img_ch, out_ch=output_ch)
        self.Decoder = UNet(in_ch=output_ch, out_ch=1)

        self.activation = torch.nn.Softmax(dim=1)

        self.Conv = nn.Conv3d(output_ch, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, ip, ip_mask=None, ops="both"):
        encoder_op = self.Encoder(ip)
        if ip_mask is not None:
            encoder_op = ip_mask * encoder_op
        class_prob = self.activation(encoder_op)
        feature_rep = self.Conv(encoder_op)
        if ops == "enc":
            return class_prob, feature_rep
        reconstructed_op = self.Decoder(class_prob)
        # if ip_mask is not None:
        #     reconstructed_op = torch.amax(ip_mask, dim=1, keepdim=True) * reconstructed_op
        if ops == "dec":
            return reconstructed_op
        if ops == "both":
            return class_prob, feature_rep, reconstructed_op
        else:
            raise ValueError('Invalid ops, ops must be in [enc, dec, both]')


class WNet3dUNetMSS(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=1, output_ch=6):
        super(WNet3dUNetMSS, self).__init__()

        self.Encoder = UNetDeepSup(in_ch=img_ch, out_ch=output_ch)
        self.Decoder = UNetDeepSup(in_ch=output_ch, out_ch=1)

        self.activation = torch.nn.Softmax(dim=1)

        self.Conv = nn.Conv3d(output_ch, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, ip, ip_mask=None, ops="both"):
        encoder_op = self.Encoder(ip)
        if ip_mask is not None:
            encoder_op = ip_mask * encoder_op
        class_prob = self.activation(encoder_op)
        feature_rep = self.Conv(encoder_op)
        if ops == "enc":
            return class_prob, feature_rep
        reconstructed_op = self.Decoder(class_prob)
        # if ip_mask is not None:
        #     reconstructed_op = torch.amax(ip_mask, dim=1, keepdim=True) * reconstructed_op
        if ops == "dec":
            return reconstructed_op
        if ops == "both":
            return class_prob, feature_rep, reconstructed_op
        else:
            raise ValueError('Invalid ops, ops must be in [enc, dec, both]')
