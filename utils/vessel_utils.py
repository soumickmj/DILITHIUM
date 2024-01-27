#!/usr/bin/env python
# from __future__ import print_function, division
"""

Purpose :

"""


import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import TiffImagePlugin
from skimage.filters import threshold_otsu
from torch.cuda.amp import GradScaler

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


def min_max(array):
    return (array - array.min()) / (array.max() - array.min())


def write_summary(writer, index, soft_ncut_loss=0, reconstruction_loss=0, mip_loss=0, total_loss=0):
    """
    Method to write summary to the tensorboard.
    index: global_index for the visualisation
    original,reconstructer: image input of dimenstion [channel, Height, Width]
    Losses: all losses used as metric
    """
    print('Writing Summary...')
    writer.add_scalar('SoftNcutLoss', soft_ncut_loss, index)
    writer.add_scalar('ReconstructionLoss', reconstruction_loss, index)
    writer.add_scalar('MIP-Loss', mip_loss, index)
    writer.add_scalar('TotalLoss', total_loss, index)


def write_epoch_summary(writer, index, soft_ncut_loss=0, reconstruction_loss=0, mip_loss=0, reg_loss=0, total_loss=0):
    """
    Method to write summary to the tensorboard.
    index: global_index for the visualisation
    Losses: all losses used as metric
    """
    print('Writing Epoch Summary...')
    writer.add_scalar('SoftNcutLossPerEpoch', soft_ncut_loss, index)
    writer.add_scalar('ReconstructionLossPerEpoch', reconstruction_loss, index)
    writer.add_scalar('MIP-LossPerEpoch', mip_loss, index)
    writer.add_scalar('RegularisationLossPerEpoch', reg_loss, index)
    writer.add_scalar('TotalLossPerEpoch', total_loss, index)


def save_model(checkpoint_path, state, filename='checkpoint', fold_index=""):
    """
    Method to save model
    """
    print('Saving model...')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(state, checkpoint_path + filename + str(state['epoch_type']) + str(fold_index) + '.pth')


def load_model(model, optimizer, checkpoint_path, batch_index='best', filename='checkpoint', fold_index=""):
    """
    Method to load model, make sure to set the model to eval, use optimiser if want to continue training
    """
    print('Loading model...')
    checkpoint = torch.load(os.path.join(checkpoint_path, filename + str(batch_index) + str(fold_index) + '.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
    return model, optimizer


def load_model_with_amp(model, optimizer, checkpoint_path, batch_index='best', filename='checkpoint', fold_index=""):
    """
    Method to load model, make sure to set the model to eval, use optimiser if want to continue training
    opt_level="O1"
    """
    print('Loading model...')
    model.cuda()
    checkpoint = torch.load(os.path.join(checkpoint_path, filename + str(batch_index) + str(fold_index) + '.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler = GradScaler()
    scaler.load_state_dict(checkpoint['amp'])
    model.eval()
    return model, optimizer, scaler


def convert_and_save_tif(image_3d, output_path, filename='output.tif', is_colored=True):
    """
    Method to convert 3D tensor to tiff image
    """
    image_list = []
    num = 3 if is_colored else 1
    for i in range(0, int(image_3d.shape[0] / num)):
        index = i * num
        tensor_image = image_3d[index:(index + num), :, :]
        image = transforms.ToPILImage(mode='RGB')(tensor_image)
        image_list.append(image)

    print('convert_and_save_tif:size of image:' + str(len(image_list)))
    with TiffImagePlugin.AppendingTiffWriter(output_path + filename, True) as tifWriter:
        for im in image_list:
            im.save(tifWriter)
            tifWriter.newFrame()
    print("Conversion to tiff completed, image saved as {}".format(filename))


def convert_and_save_tif_greyscale(image_3d, output_path, filename='output.tif'):
    """
    Method to convert 3D tensor to tiff image
    """
    image_list = []

    for i in range(0, int(image_3d.shape[0])):
        tensor_image = image_3d[i]
        image = transforms.ToPILImage(mode='F')(tensor_image)
        image_list.append(image)

    print('convert_and_save_tif:size of image:' + str(len(image_list)))
    with TiffImagePlugin.AppendingTiffWriter(output_path + filename, True) as tifWriter:
        for im in image_list:
            # with open(DATASET_FOLDER+tiff_in) as tiff_in:
            im.save(tifWriter)
            tifWriter.newFrame()
    print("Conversion to tiff completed, image saved as {}".format(filename))


def create_mask(predicted, logger):
    """
    Method find the difference between the 2 images and overlay colors
    predicted, label : slices , 2D tensor
    """
    predicted = predicted.cpu().data.numpy()

    try:
        thresh = threshold_otsu(predicted)
        predicted_binary = predicted > thresh
    except Exception as error:
        logger.exception(error)
        predicted_binary = predicted > 0.5  # exception will be thrown if input image seems to have just one color 1.0.

    # Define colors
    black = np.array([0, 0, 0], dtype=np.uint8)  # background
    white = np.array([255, 255, 255], dtype=np.uint8)  # prediction_output

    # Make RGB array, pre-filled with black(background)
    rgb_image = np.zeros((predicted_binary.shape[0], predicted_binary.shape[1], 3), dtype=np.uint8) + black

    # Overwrite with red where threshold exceeded, i.e. where mask is True
    rgb_image[predicted_binary] = white

    return rgb_image


def create_diff_mask(predicted, label, logger):
    """
    Method find the difference between the 2 images(predicted being grescale, label being binary) and overlay colors
    predicted, label : slices , 2D tensor
    """
    label = label.cpu().data.numpy()
    predicted = predicted.cpu().data.numpy()

    try:
        thresh = threshold_otsu(predicted)
        predicted_binary = predicted > thresh
    except Exception as error:
        logger.exception(error)
        predicted_binary = predicted > 0.5  # exception will be thrown if input image seems to have just one color 1.0.

    diff1 = np.subtract(label, predicted_binary) > 0
    diff2 = np.subtract(predicted_binary, label) > 0

    # Define colors
    red = np.array([255, 0, 0], dtype=np.uint8)  # under_detected
    # green = np.array([0, 255, 0], dtype=np.uint8)  # over_detected
    black = np.array([0, 0, 0], dtype=np.uint8)  # background
    white = np.array([255, 255, 255], dtype=np.uint8)  # prediction_output
    blue = np.array([0, 0, 255], dtype=np.uint8)  # over_detected
    # yellow = np.array([255, 255, 0], dtype=np.uint8)  # under_detected

    # Make RGB array, pre-filled with black(background)
    rgb_image = np.zeros((predicted_binary.shape[0], predicted_binary.shape[1], 3), dtype=np.uint8) + black

    # Overwrite with red where threshold exceeded, i.e. where mask is True
    rgb_image[predicted_binary] = white
    rgb_image[diff1] = red
    rgb_image[diff2] = blue

    return rgb_image


def create_diff_mask_binary(predicted, label):
    """
    Method find the difference between the 2 binary images and overlay colors
    predicted, label : slices , 2D tensor
    """
    predicted_label = label.cpu().data.numpy()
    predicted_binary = predicted.cpu().data.numpy()

    diff1 = np.subtract(predicted_label, predicted_binary) > 0
    diff2 = np.subtract(predicted_binary, predicted_label) > 0

    predicted_binary = predicted_binary > 0

    # Define colors
    red = np.array([255, 0, 0], dtype=np.uint8)  # under_detected
    # green = np.array([0, 255, 0], dtype=np.uint8)  # over_detected
    black = np.array([0, 0, 0], dtype=np.uint8)  # background
    white = np.array([255, 255, 255], dtype=np.uint8)  # prediction_output
    blue = np.array([0, 0, 255], dtype=np.uint8)  # over_detected
    # yellow = np.array([255, 255, 0], dtype=np.uint8)  # under_detected

    # Make RGB array, pre-filled with black(background)
    rgb_image = np.zeros((predicted_binary.shape[0], predicted_binary.shape[1], 3), dtype=np.uint8) + black

    # Overwrite with red where threshold exceeded, i.e. where mask is True
    rgb_image[predicted_binary] = white
    rgb_image[diff1] = red
    rgb_image[diff2] = blue
    return rgb_image


def show_diff(label, predicted, diff_image):
    """
    Method to display the differences between label, predicted and diff_image
    """
    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[0].imshow(label, cmap=plt.cm.gray)
    ax[0].set_title('GroundTruth')
    ax[0].axis('off')

    ax[1].imshow(predicted, cmap=plt.cm.gray)
    ax[1].set_title('Predicted')
    ax[0].axis('off')

    ax[2].imshow(diff_image, cmap=plt.cm.gray)
    ax[2].set_title('Difference image')
    ax[2].axis('off')

    plt.show()
