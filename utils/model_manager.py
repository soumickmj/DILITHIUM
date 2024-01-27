#!/usr/bin/env python
"""
Purpose : model selector
"""

from models.w_net_3d import WNet3dUNet, WNet3dAttUNet, WNet3dUNetMSS


__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

MODEL_UNET = 1
MODEL_UNET_DEEPSUP = 2
MODEL_ATTENTION_UNET = 3


def get_model(model_no, output_ch):  # Send model params from outside
    default_model = WNet3dAttUNet(output_ch=output_ch)  # Default
    model_list = {
        1: WNet3dUNet(output_ch=output_ch),
        2: WNet3dUNetMSS(output_ch=output_ch),
        3: WNet3dAttUNet(output_ch=output_ch),
    }
    return model_list.get(model_no, default_model)
