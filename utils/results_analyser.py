import os
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, TiffImagePlugin


def create_diff_mask_binary(predicted, label):
    """
    Method find the difference between the 2 binary images and overlay colors
    predicted, label : slices , 2D tensor
    """

    diff1 = np.subtract(label, predicted) > 0 # under_detected
    diff2 = np.subtract(predicted, label) > 0 # over_detected

    predicted = predicted > 0

    # Define colors
    red = np.array([255, 0, 0], dtype=np.uint8)  # under_detected
    green = np.array([0, 255, 0], dtype=np.uint8)  # over_detected
    black = np.array([0, 0, 0], dtype=np.uint8)  # background
    white = np.array([255, 255, 255], dtype=np.uint8)  # prediction_output
    blue = np.array([0, 0, 255], dtype=np.uint8)  # over_detected
    yellow = np.array([255, 255, 0], dtype=np.uint8)  # under_detected

    # Make RGB array, pre-filled with black(background)
    rgb_image = np.zeros((*predicted.shape, 3), dtype=np.uint8) + black

    # Overwrite with red where threshold exceeded, i.e. where mask is True
    rgb_image[predicted] = white
    rgb_image[diff1] = red
    rgb_image[diff2] = blue
    return rgb_image


def create_segmentation_mask(predicted, colors, num_classes=3):
    """
        Method creates a segmentation mask for identified classes for the class predictions
        predicted : 2D mip with class predictions
        """

    # Make RGB array, pre-filled with black(background)
    rgb_image = np.zeros((*predicted.shape, 3), dtype=np.uint8) + colors['background']

    # Overwrite with red where threshold exceeded, i.e. where mask is True
    for segment in range(num_classes):
        segment_mask = np.where(predicted == segment)
        if segment_mask is not None:
            rgb_image[segment_mask] = colors['segment'+str(segment)]

    return rgb_image


def save_color_nifti(vol3d, output_path):
    shape_3d = vol3d.shape[0:3]
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    vol3d = vol3d.copy().view(dtype=rgb_dtype).reshape(shape_3d)  # copy used to force fresh internal structure
    ni_img = nib.Nifti1Image(vol3d, np.eye(4))
    nib.save(ni_img, output_path)


def save_nifti(vol, path):
    img = nib.Nifti1Image(vol, np.eye(4))
    nib.save(img, path)


def save_nifti_rgb(vol, path):
    shape_3d = vol.shape[0:3]
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    vol = vol.copy().view(dtype=rgb_dtype).reshape(shape_3d)  # copy used to force fresh internal structure
    nii_img = nib.Nifti1Image(vol, np.eye(4))
    nib.save(nii_img, path)


def contains_colour(s, tp):
    for t in tp:
        if s == t[1]:
            return True


def save_tif_rgb(image3d, output_path):
    """
    Method to convert 3D tensor to tiff image
    """
    image_list = []
    for i in range(0, image3d.shape[-2]):
        image = image3d[..., i, :]
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        # if contains_colour((255,255,255),image.getcolors()):
        #     sds
        image_list.append(image)

    with TiffImagePlugin.AppendingTiffWriter(output_path, True) as tifWriter:
        for im in image_list:
            # with open(DATASET_FOLDER+tiff_in) as tiff_in:
            im.save(tifWriter)
            tifWriter.newFrame()


def dice(pred, true, k=1):
    intersection = np.sum(pred[true == k]) * 2.0
    dice_loss = intersection / (np.sum(pred) + np.sum(true))
    return dice_loss


def iou(pred, true):
    intersection = np.logical_and(pred, true)
    union = np.logical_or(pred, true)
    return np.sum(intersection) / np.sum(union)


def score_slabs(pred, true, dim=-1, n_slabs=-1):
    if n_slabs == -1:  # 2D
        n_slabs = pred.shape[dim]
    slabs_pred = np.array_split(pred, n_slabs, dim)
    slabs_true = np.array_split(true, n_slabs, dim)
    slabs_dice = np.zeros(len(slabs_pred))
    slabs_iou = np.zeros(len(slabs_pred))
    for i in range(len(slabs_pred)):
        slabs_dice[i] = dice(slabs_pred[i], slabs_true[i])
        slabs_iou[i] = iou(slabs_pred[i], slabs_true[i])
    return slabs_dice, slabs_iou, list(range(len(slabs_pred)))


def score_results(result_folder, labels_folder):
    df = pd.DataFrame(columns=["Subject", "Dice", "IoU"])

    results = glob(result_folder + "/*.nii") + glob(result_folder + "/*.nii.gz")
    labels = glob(labels_folder + "/*.nii") + glob(labels_folder + "/*.nii.gz")
    subjects = []
    for i in range(len(results)):
        r = results[i]
        filename = os.path.basename(r).split('.')[0]
        l = [s for s in labels if filename in s][0]

        result = nib.load(r).get_fdata()
        label = nib.load(l).get_fdata()

        datum = {"Subject": filename}
        dice3d = dice(result, label)
        iou3d = iou(result, label)
        datum = pd.DataFrame.from_dict({**datum, "Dice": [dice3d], "IoU": [iou3d]})
        df = pd.concat([df, datum], ignore_index=True)

    df.to_excel(os.path.join(res_folder, "Results_Main.xlsx"))


if __name__ == '__main__':
    res_folder = "/home/schatter/Soumick/Output/DS6/OrigVol_MaskedFDIPv0_ProbUNet_AMP_NoGradClip/results"
    label_folder = "/vol3/schatter/DS6/Dataset/BiasFieldCorrected/300/test_label"
    score_results(res_folder, label_folder)
