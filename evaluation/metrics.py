#!/usr/bin/env python

"""

Purpose :

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
# from torchmetrics.functional import structural_similarity_index_measure
import numpy as np
import torchio as tio
# from pytorch_msssim import SSIM
from .perceptual_loss import PerceptualLoss
from scipy.stats import norm

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class SoftNCutsLoss_v1(nn.Module):
    def __init__(self, depth, length, width, std_position=1):
        super(SoftNCutsLoss_v1, self).__init__()
        meshgrid_x, meshgrid_y, meshgrid_z = torch.meshgrid(torch.arange(0, depth, dtype=float),
                                                            torch.arange(0, length, dtype=float),
                                                            torch.arange(0, width, dtype=float))
        meshgrid_x = torch.reshape(meshgrid_x, (length * width * depth,))
        meshgrid_y = torch.reshape(meshgrid_y, (length * width * depth,))
        meshgrid_z = torch.reshape(meshgrid_z, (length * width * depth,))
        A_x = SoftNCutsLoss_v1._outer_product(meshgrid_x, torch.ones(meshgrid_x.size(), dtype=meshgrid_x.dtype,
                                                                     device=meshgrid_x.device))
        A_y = SoftNCutsLoss_v1._outer_product(meshgrid_y, torch.ones(meshgrid_y.size(), dtype=meshgrid_y.dtype,
                                                                     device=meshgrid_y.device))
        A_z = SoftNCutsLoss_v1._outer_product(meshgrid_z, torch.ones(meshgrid_z.size(), dtype=meshgrid_z.dtype,
                                                                     device=meshgrid_z.device))

        del meshgrid_x, meshgrid_y, meshgrid_z

        xi_xj = A_x - A_x.permute(1, 0)
        yi_yj = A_y - A_y.permute(1, 0)
        zi_zj = A_z - A_z.permute(1, 0)

        sq_distance_matrix = torch.square(xi_xj) + torch.square(yi_yj) + torch.square(zi_zj)

        del A_x, A_y, A_z, xi_xj, yi_yj, zi_zj

        self.dist_weight = torch.exp(-torch.divide(sq_distance_matrix, torch.square(torch.tensor(std_position))))

    @staticmethod
    def _outer_product(v1, v2):
        """
        Inputs:
        v1 : m*1 tf array
        v2 : m*1 tf array
        Output :
        v1 x v2 : m*m array
        """
        v1 = torch.reshape(v1, (-1,))
        v2 = torch.reshape(v2, (-1,))
        v1 = torch.unsqueeze(v1, 0)
        v2 = torch.unsqueeze(v2, 0)
        return torch.matmul(v1.T, v2)

    def _edge_weights(self, flatten_patch, std_intensity=3):
        """
        Inputs :
        flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
        std_intensity : standard deviation for intensity
        std_position : standard devistion for position
        radius : the length of the around the pixel where the weights
        is non-zero
        rows : rows of the original image (unflattened image)
        cols : cols of the original image (unflattened image)
        Output :
        weights :  2d tf array edge weights in the pixel graph
        Used parameters :
        n : number of pixels
        """
        A = SoftNCutsLoss_v1._outer_product(flatten_patch, torch.ones_like(flatten_patch))
        intensity_weight = torch.exp(-1 * torch.square((torch.divide((A - A.T), std_intensity)))).detach().cpu()
        del A
        return torch.multiply(intensity_weight, self.dist_weight)

    @staticmethod
    def _numerator(k_class_prob, weights):
        """
        Inputs :
        k_class_prob : k_class pixelwise probability (rows*cols) tensor
        weights : edge weights n*n tensor
        """
        k_class_prob = torch.reshape(k_class_prob, (-1,))
        return torch.sum(torch.multiply(weights, SoftNCutsLoss_v1._outer_product(k_class_prob, k_class_prob)))

    @staticmethod
    def _denominator(k_class_prob, weights):
        """
        Inputs:
        k_class_prob : k_class pixelwise probability (rows*cols) tensor
        weights : edge weights	n*n tensor
        """
        k_class_prob = torch.reshape(k_class_prob, (-1,))
        return torch.sum(torch.multiply(weights, SoftNCutsLoss_v1._outer_product(k_class_prob,
                                                                                 torch.ones(k_class_prob.size(),
                                                                                            dtype=k_class_prob.dtype,
                                                                                            layout=k_class_prob.layout,
                                                                                            device=k_class_prob.device))))

    def forward(self, patch, prob, k):
        """
        Inputs:
        prob : (rows*cols*k) tensor
        k : number of classes (integer)
        flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
        rows : number of the rows in the original image
        cols : number of the cols in the original image
        Output :
        soft_n_cut_loss tensor for a single image
        """
        flatten_patch = torch.flatten(patch)
        soft_n_cut_loss = k
        weights = self._edge_weights(flatten_patch)
        prob = prob.cpu()

        for t in range(k):
            soft_n_cut_loss = soft_n_cut_loss - (
                    SoftNCutsLoss_v1._numerator(prob[:, :, :, t], weights) / SoftNCutsLoss_v1._denominator(
                prob[:, :, :, t],
                weights))

        del weights
        del flatten_patch
        return soft_n_cut_loss.float().cuda()


class SoftNCutsLoss_v2(nn.Module):
    def __init__(self, radius=4, sigmaI=10, sigmaX=4, num_classes=8, batch_size=15, patch_size=32):
        super(SoftNCutsLoss_v2, self).__init__()
        self.radius = radius
        self.sigmaI = sigmaI
        self.sigmaX = sigmaX
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.ip_shape = (batch_size, 1, patch_size, patch_size, patch_size)
        self.pad = torch.nn.ConstantPad3d(radius - 1, np.finfo(np.float).eps)
        self.dissim_matrix = torch.zeros(
            (self.ip_shape[1], self.ip_shape[2], self.ip_shape[3], self.ip_shape[4],
             (radius - 1) * 2 + 1,
             (radius - 1) * 2 + 1, (radius - 1) * 2 + 1))
        self.dist = torch.zeros((2 * (self.radius - 1) + 1, 2 * (self.radius - 1) + 1, 2 * (self.radius - 1) + 1))
        self.cropped_seg = torch.zeros((num_classes, patch_size, patch_size, patch_size,
                                        2 * (self.radius - 1) + 1, 2 * (self.radius - 1) + 1,
                                        2 * (self.radius - 1) + 1))

    def _cal_weights(self, batch, padded_batch):
        """
        Inputs:
        batch : ip batch (B x C x D x H x W)
        padded_batch : padded ip batch
        Output :
        weight and sum weight
        """
        # According to the weight formula, when Euclidean distance < r,the weight is 0,
        # so reduce the dissim matrix size to radius-1 to save time and space.
        # print("calculating weights.")
        temp_dissim = self.dissim_matrix.expand(batch.shape[0], -1, -1, -1, -1, -1, -1, -1).clone().cuda()
        for x in range(2 * (self.radius - 1) + 1):
            for y in range(2 * (self.radius - 1) + 1):
                for z in range(2 * (self.radius - 1) + 1):
                    temp_dissim[:, :, :, :, :, x, y, z] = batch - padded_batch[:, :, x:self.patch_size + x,
                                                                  y:self.patch_size + y, z:self.patch_size + z]

        temp_dissim = torch.exp(-1 * torch.square(temp_dissim) / self.sigmaI ** 2)
        temp_dist = self.dist.clone().cuda()
        for x in range(1 - self.radius, self.radius):
            for y in range(1 - self.radius, self.radius):
                for z in range(1 - self.radius, self.radius):
                    if x ** 2 + y ** 2 + z ** 2 < self.radius ** 2:
                        temp_dist[x + self.radius - 1, y + self.radius - 1, z + self.radius - 1] = np.exp(
                            -(x ** 2 + y ** 2 + z ** 2) / self.sigmaX ** 2)

        weight = torch.multiply(temp_dissim, temp_dist)
        del temp_dissim, temp_dist
        sum_weight = weight.sum(-1).sum(-1).sum(-1)
        return weight, sum_weight

    def forward(self, batch, preds):
        """
        Inputs:
        patch : ip patch (B x C x D x H x W)
        preds : class predictions (B x K x D x H x W)
        Output :
        soft_n_cut_loss tensor for a batch of ip patch and K-class predictions
        """
        padded_preds = self.pad(preds)
        # According to the weight formula, when Euclidean distance < r,
        # the weight is 0, so reduce the dissim matrix size to radius-1 to save time and space.
        padded_batch = self.pad(batch)
        weight, sum_weight = self._cal_weights(batch=batch, padded_batch=padded_batch)

        # too many values to unpack
        temp_cropped_seg = self.cropped_seg.expand(preds.shape[0], -1, -1, -1, -1, -1, -1, -1).clone().cuda()
        for x in range((self.radius - 1) * 2 + 1):
            for y in range((self.radius - 1) * 2 + 1):
                for z in range((self.radius - 1) * 2 + 1):
                    temp_cropped_seg[:, :, :, :, :, x, y, z] = padded_preds[:, :, x:x + preds.size()[2],
                                                               y:y + preds.size()[3],
                                                               z:z + preds.size()[4]]
        # cropped_seg = []
        # for x in torch.arange((self.radius - 1) * 2 + 1, dtype=torch.long):
        #     width = []
        #     for y in torch.arange((self.radius - 1) * 2 + 1, dtype=torch.long):
        #         depth = []
        #         for z in torch.arange((self.radius - 1) * 2 + 1, dtype=torch.long):
        #             depth.append(
        #                 padded_preds[:, :, x:x + preds.size()[2], y:y + preds.size()[3], z:z + preds.size()[4]])
        #         width.append(torch.stack(depth, 5))
        #     cropped_seg.append(torch.stack(width, 5))
        # cropped_seg = torch.stack(cropped_seg, 5)

        numerator = temp_cropped_seg.mul(weight)
        del temp_cropped_seg
        numerator = numerator.sum(-1).sum(-1).sum(-1).mul(preds)
        denominator = sum_weight.mul(preds)

        assocA = numerator.view(numerator.shape[0], numerator.shape[1], -1).sum(-1)
        assocV = denominator.view(denominator.shape[0], denominator.shape[1], -1).sum(-1)
        assoc = assocA.div(assocV).sum(-1)

        soft_n_cut_loss = torch.add(-assoc, self.num_classes)
        return soft_n_cut_loss


class SoftNCutsLoss_v3(nn.Module):
    def __init__(self, radius=4, sigma_x=4, sigma_i=10, patch_size=32, n_channel=1):
        super(SoftNCutsLoss_v3, self).__init__()
        self.radius = radius
        self.sigma_x = sigma_x
        self.sigma_i = sigma_i
        self.patch_size = patch_size
        self.n_channel = n_channel
        self.pad = torch.nn.ConstantPad3d(radius, np.finfo(np.float).eps)
        self.neighborhood_size = radius * 2 + 1
        dist_dim = (torch.arange(1, self.neighborhood_size + 1) - torch.arange(1, self.neighborhood_size + 1)[radius]) \
            .expand(self.neighborhood_size, self.neighborhood_size, self.neighborhood_size)
        dist3d = 3 * (dist_dim ** 2)
        dist_mask = dist3d.le(radius)
        dist_weights = torch.exp(torch.div(-1 * dist3d, sigma_x ** 2))
        self.dist_weights = torch.mul(dist_mask, dist_weights)

    def _cal_weights(self, batch):
        padded_batch = self.pad(batch)
        padded_dissim = padded_batch.unfold(2, self.radius * 2 + 1, 1) \
            .unfold(3, self.radius * 2 + 1, 1) \
            .unfold(4, self.radius * 2 + 1, 1)
        padded_dissim = padded_dissim.contiguous().view(batch.shape[0], self.n_channel, -1,
                                                        self.neighborhood_size, self.neighborhood_size,
                                                        self.neighborhood_size)
        padded_dissim = padded_dissim.permute(0, 2, 1, 3, 4, 5)
        padded_dissim = padded_dissim.view(-1, self.n_channel,
                                           self.neighborhood_size, self.neighborhood_size, self.neighborhood_size)
        center_values = padded_dissim[:, :, self.radius, self.radius, self.radius]
        center_values = center_values[:, :, None, None, None]
        center_values = center_values.expand(-1, -1,
                                             self.neighborhood_size, self.neighborhood_size, self.neighborhood_size)
        padded_dissim = torch.exp(torch.div(-1 * ((padded_dissim - center_values) ** 2), self.sigma_i ** 2))
        return torch.mul(padded_dissim, self.dist_weights.clone().cuda())

    def _cal_loss_for_k(self, weights, preds, batch_size):
        padded_preds = self.pad(preds)
        pred_seg = padded_preds.unfold(2, self.neighborhood_size, 1) \
            .unfold(3, self.neighborhood_size, 1) \
            .unfold(4, self.neighborhood_size, 1)
        pred_seg = pred_seg.contiguous().view(batch_size, self.n_channel, -1,
                                              self.neighborhood_size, self.neighborhood_size, self.neighborhood_size)
        pred_seg = pred_seg.permute(0, 2, 1, 3, 4, 5)
        pred_seg = pred_seg.view(-1, self.n_channel,
                                 self.neighborhood_size, self.neighborhood_size, self.neighborhood_size)
        numerator = weights * pred_seg
        numerator = torch.sum(preds * torch.sum(numerator, dim=(1, 2, 3, 4))
                              .reshape(batch_size, self.patch_size, self.patch_size, self.patch_size), dim=(1, 2, 3, 4))
        denominator = torch.sum(preds * torch.sum(weights, dim=(1, 2, 3, 4))
                                .reshape(batch_size, self.patch_size, self.patch_size, self.patch_size),
                                dim=(1, 2, 3, 4))
        return torch.div(numerator + np.finfo(np.float).eps, denominator + np.finfo(np.float).eps)

    def forward(self, batch, preds):
        batch_size = batch.shape[0]
        k = preds.shape[1]  # num classes
        weights = self._cal_weights(batch)
        loss = [self._cal_loss_for_k(weights, preds[:, (i,), :, :, :], batch_size) for i in range(k)]
        total_loss = torch.stack(loss)
        return k - torch.sum(total_loss, dim=0)


class SoftNCutsLoss(nn.Module):
    r"""Implementation of the continuous N-Cut loss, as in:
    'W-Net: A Deep Model for Fully Unsupervised Image Segmentation', by Xia, Kulis (2017)"""

    def __init__(self, radius=4, sigma_x=5, sigma_i=1):
        r"""
        :param radius: Radius of the spatial interaction term
        :param sigma_1: Standard deviation of the spatial Gaussian interaction
        :param sigma_2: Standard deviation of the pixel value Gaussian interaction
        """
        super(SoftNCutsLoss, self).__init__()
        self.radius = radius
        self.sigma_x = sigma_x  # Spatial standard deviation
        self.sigma_i = sigma_i  # Pixel value standard deviation

    def gaussian_kernel3d(self):
        neighborhood_size = 2 * self.radius + 1
        voxel_neighborhood = np.linspace(-self.radius, self.radius, neighborhood_size) ** 2
        xy, yz, zx = np.meshgrid(voxel_neighborhood, voxel_neighborhood, voxel_neighborhood)
        dist = (xy + yz + zx) / self.sigma_x
        kernel = norm.pdf(dist) / norm.pdf(0)
        kernel = torch.from_numpy(kernel.astype(np.float32))
        kernel = kernel.view((1, 1, kernel.shape[0], kernel.shape[1], kernel.shape[2]))
        kernel = kernel.cuda()
        return kernel

    def forward(self, inputs, labels):
        r"""Computes the continuous N-Cut loss, given a set of class probabilities (labels) and raw images (inputs).
        Small modifications have been made here for efficiency -- specifically, we compute the pixel-wise weights
        relative to the class-wide average, rather than for every individual pixel.
        :param labels: Predicted class probabilities
        :param inputs: Raw images
        :return: Continuous N-Cut loss
        """
        num_classes = labels.shape[1]
        loss = 0
        kernel = self.gaussian_kernel3d()

        for k in range(num_classes):
            # Compute the average pixel value for this class, and the difference from each pixel
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(inputs * class_probs, dim=(2, 3, 4), keepdim=True) / \
                         torch.add(torch.mean(class_probs, dim=(2, 3, 4), keepdim=True), 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)

            # Weight the loss by the difference from the class average.
            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_i ** 2))

            # Compute N-cut loss, using the computed weights matrix, and a Gaussian spatial filter
            numerator = torch.sum(class_probs * F.conv3d(class_probs * weights, kernel, padding=self.radius))
            denominator = torch.sum(class_probs * F.conv3d(weights, kernel, padding=self.radius))
            loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6), torch.zeros_like(numerator))

        return num_classes - loss


class ReconstructionLoss(nn.Module):
    def __init__(self, recr_loss_model_path=None, loss_type="L1"):
        super(ReconstructionLoss, self).__init__()
        self.loss = PerceptualLoss(loss_model="unet3Dds", model_load_path=recr_loss_model_path, loss_type=loss_type)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


def l2_regularisation_loss(model):
    l2_reg = torch.tensor(0.0, requires_grad=True)

    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_reg = l2_reg + param.norm(2)
    return l2_reg


class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1, gamma=0.75, alpha=0.7):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        pt_1 = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        return pow(abs(1 - pt_1), self.gamma)
