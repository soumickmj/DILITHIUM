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


def load_model_from_hf(repo_id):
    """
    Load a pretrained WNet model from a HuggingFace Hub repository.
    The repository must use the AutoModel/trust_remote_code pattern
    (i.e. contain WNets.py, WNetConfigs.py, and model.safetensors).
    Returns the inner PyTorch model (WNet3dUNet / WNet3dAttUNet / WNet3dUNetMSS)
    with pretrained weights already loaded.
    """
    from transformers import AutoModel
    hf_model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    return hf_model.model
