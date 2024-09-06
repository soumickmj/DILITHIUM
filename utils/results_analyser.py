import os
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, TiffImagePlugin
from scipy import ndimage, spatial


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
    """
    Purpose: save 3D RGB nibabel nifti image
    :param vol3d: 3D array containing image data
    :param output_path: Path to target image save location
    """
    shape_3d = vol3d.shape[0:3]
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    vol3d = vol3d.copy().view(dtype=rgb_dtype).reshape(shape_3d)  # copy used to force fresh internal structure
    ni_img = nib.Nifti1Image(vol3d, np.eye(4))
    nib.save(ni_img, output_path)


def save_nifti(vol, path):
    """
    Purpose: save 3D nibabel nifti image
    :param vol: 3D array containing image data
    :param path: Path to target image save location
    """
    img = nib.Nifti1Image(vol, np.eye(4))
    nib.save(img, path)


def save_nifti_rgb(vol, path):
    """
    Purpose: save 3D RGB nibabel nifti image
    :param vol: 3D array containing image data
    :param path: Path to target image save location
    """
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
    """
    Purpose: Method to compute dice score between two 3D arrays
    """
    intersection = np.sum(pred[true == k]) * 2.0
    dice_loss = intersection / (np.sum(pred) + np.sum(true))
    return dice_loss


def iou(pred, true):
    """
    Purpose: Method to compute iou between two 3D arrays
    """
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


def calculate_vessel_diameter3d(vessel_mask):
    """
    Purpose: Calculate the diameter(s) of the underlying vessels in the input segmentation in 3D
    :param vessel_mask: input segmentation
    """
    # Compute the distance matrix (same size as the input)
    # where the pixel values (0 or 1) are replaced by its shortest distance to a zero voxel
    distance_transform_dist, distance_transform_ids = ndimage.morphology.distance_transform_edt(vessel_mask.astype(np.uint8), return_indices=True)
    distance_transform_dist = distance_transform_dist.astype(np.uint8)

    # Generate a KD Tree for traversing the nearest neighbors of each voxel
    x_len, y_len, z_len = vessel_mask.shape
    x, y, z = np.mgrid[0:x_len, 0:y_len, 0:z_len]
    points = np.c_[x.ravel(), y.ravel(), z.ravel()]
    tree = spatial.KDTree(points)

    # create a diameter_map of same size as input
    diameter_map = np.zeros_like(vessel_mask)
    # points = np.asarray(points)

    # For each voxel in the MIP, calculate the maximum diameter based on its neighboring voxel
    list(map(lambda coords: estimate_diameter3d(vessel_mask, diameter_map, tree, points, coords, distance_transform_dist), np.ndindex(vessel_mask.shape)))

    return diameter_map


def estimate_diameter3d(vessel_mask, diameter_map, tree, points, coords, distance_transform_dist):
    """
    Purpose: Calculate the maximum diameter of each voxel (in 3D) based on its neighbours
    :param vessel_mask: input segmentation
    :param diameter_map: an array of zeros same size as the input segmentation
    which is updated with the maximum diameter for each voxel
    :param tree: a KD neighbourhood traversal tree
    :param points: voxels on the 3D meshgrid for the tree traversal
    :param coords: voxel co-ordinates (x, y, z)
    :param distance_transform_dist: distance matrix for the input segmentation obtained by applying morphological distance transform
    """
    neighbors = []
    neighbor_dist = []
    diameter = 0

    # For each neighbor of a non-zero voxel,
    # if the neighbor is a non-zero voxel then, collect its shortest distance to a zero voxel
    for results in tree.query_ball_point(coords, 1):
        if vessel_mask[coords] == 0:
            continue
        ids = points[results]
        if vessel_mask[ids[0], ids[1], ids[2]] == 1:
            neighbors.append(points[results])
            neighbor_dist.append(distance_transform_dist[ids[0], ids[1], ids[2]])
            # diameter += 1

    # If the pixel has non-zero neighbors then,
    # assign the maximum distance to zero voxel as the diameter of the voxel
    if len(neighbor_dist) > 0:
        diameter = np.max(neighbor_dist)
    diameter_map[coords] = diameter

    return coords, neighbors, diameter


def calculate_vessel_diameter2d(vessel_mask):
    """
    Purpose: Calculate the diameter(s) of the underlying vessels in the input segmentation in 2D
    :param vessel_mask: input segmentation
    """
    # Compute the MIP of the segmentation mask
    mask_mip = np.max(vessel_mask, axis=-1)

    # Compute the distance matrix (same size as the input)
    # where the pixel values (0 or 1) are replaced by its shortest distance to a zero pixel
    distance_transform_dist, distance_transform_ids = ndimage.morphology.distance_transform_edt(mask_mip.astype(np.uint8), return_indices=True)
    distance_transform_dist = distance_transform_dist.astype(np.uint8)

    # Generate a KD Tree for traversing the nearest neighbors of each pixel
    x_len, y_len = mask_mip.shape
    x, y = np.mgrid[0:x_len, 0:y_len]
    points = np.c_[x.ravel(), y.ravel()]
    tree = spatial.KDTree(points)

    # create a diameter_map of same size as input
    diameter_map = np.zeros_like(mask_mip)
    # points = np.asarray(points)

    # For each pixel in the MIP, calculate the maximum diameter based on its neighboring pixels
    list(map(lambda coords: estimate_diameter2d(mask_mip, diameter_map, tree, points, coords, distance_transform_dist), np.ndindex(mask_mip.shape)))

    return diameter_map


def estimate_diameter2d(mask_mip, diameter_map, tree, points, coords, distance_transform_dist):
    """
    Purpose: Calculate the maximum diameter of each pixel (in 2D) based on its neighbours
    :param mask_mip: 2D MIP of the 3D input segmentation
    :param diameter_map: an array of zeros same size as the input segmentation
    which is updated with the maximum diameter for each pixel
    :param tree: a KD neighbourhood traversal tree
    :param points: pixels on the 2D meshgrid for the tree traversal
    :param coords: pixel co-ordinates (x, y)
    :param distance_transform_dist: distance matrix for the input segmentation obtained by applying morphological distance transform
    """
    neighbors = []
    neighbor_dist = []
    diameter = 0

    # For each neighbor of a non-zero pixel,
    # if the neighbor is a non-zero pixel then, collect its shortest distance to a zero pixel
    for results in tree.query_ball_point(coords, 1):
        if mask_mip[coords] == 0:
            continue
        ids = points[results]
        if mask_mip[ids[0], ids[1]] == 1:
            neighbors.append(points[results])
            neighbor_dist.append(distance_transform_dist[ids[0], ids[1]])
            # diameter += 1

    # If the pixel has non-zero neighbors then,
    # assign the maximum distance to zero pixel as the diameter of the pixel
    if len(neighbor_dist) > 0:
        diameter = np.max(neighbor_dist)
    diameter_map[coords] = diameter

    return coords, neighbors, diameter


def save_mask(mask, output_path, op_filename, dim=2):
    """
    Purpose: Save resulting 2D diameter mask as png or 3D diameter mask as .nii.gz
    """
    if dim == 2:
        out_img = Image.fromarray(mask.astype('uint8'))
        out_img.save(os.path.join(output_path, op_filename + ".png"))
    else:
        img = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
        nib.save(img, os.path.join(output_path, op_filename + ".nii.gz"))
    print("Saved " + op_filename + " successfully!!")


def diameter_mask_ranking(pred_mask, gt_mask, output_path, subject_name, dim, save_img=False):
    """
    Purpose: Rank different underlying vessels in the input segmentation based on the calculated vessel diameters.
    """
    dice_df = []
    vessel_diameters = np.unique(pred_mask)
    for diameter in vessel_diameters:
        if diameter == 0:
            continue
        mask_dm_binary = pred_mask == diameter
        gt_mask_binary = gt_mask == diameter
        dice_score = dice(mask_dm_binary, gt_mask_binary)
        dice_df.append({"Method": subject_name, "Diameter": diameter, "Dice": dice_score})
        if save_img:
            save_mask(mask_dm_binary.astype(np.uint8), output_path, subject_name + "_pred_" + str(diameter), dim=dim)
            save_mask(gt_mask_binary.astype(np.uint8), output_path, subject_name + "_gt_" + str(diameter), dim=dim)
    return dice_df


if __name__ == '__main__':
    res_folder = "/home/schatter/Soumick/Output/DS6/OrigVol_MaskedFDIPv0_ProbUNet_AMP_NoGradClip/results"
    label_folder = "/vol3/schatter/DS6/Dataset/BiasFieldCorrected/300/test_label"
    score_results(res_folder, label_folder)
