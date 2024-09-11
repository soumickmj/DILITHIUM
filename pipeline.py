# !/usr/bin/env python
"""
Purpose: 3D unsupervised and weakly-supervised patch-based vessel segmentation pipeline implementation.
Features:
* Baseline WNet training and testing
* Training unsupervised Att. WNet, weakly-supervised Att. WNet with MIP (single and multi-axial) loss.
* Training unsupervised MSS WNet, weakly-supervised MSS WNet with MIP (single and multi-axial) loss.
* Extracting inference and post-processing of the predicted segmentation
"""

import torch
import torch.utils.data
import torchio as tio
# from PIL import Image
# from torchviz import make_dot
from skimage.filters import threshold_otsu
from skimage.morphology import area_opening, ball, dilation
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sys

from evaluation.metrics import (SoftNCutsLoss, ReconstructionLoss, l2_regularisation_loss, FocalTverskyLoss)
from utils.datasets import SRDataset
from utils.madam import Madam
from utils.mtadam import MTAdam
# from torchmetrics.functional import structural_similarity_index_measure
# from pytorch_msssim import ssim
from utils.results_analyser import *
from utils.vessel_utils import (load_model, load_model_with_amp, save_model, write_epoch_summary)

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class Pipeline:

    def __init__(self, cmd_args, model, logger, dir_path, checkpoint_path, writer_training, writer_validating,
                 wandb=None):
        """
        :param cmd_args: command line arguments for initializing network parameters, experimental hyper-parameters and
        environmental parameters
        :param model: initialized baseline network i.e., 1: UNet, 2: UNet-MSS, 3: Attention UNet
        :param logger: File logger
        :param dir_path: Dataset Folder location
        :param checkpoint_path: Path to saved model state dictionary
        :param writer_training: Initialized train writer
        :param writer_validating: Initialized validation writer
        :param wandb: Initialized 'Weights and Biases' configuration for logging
        """
        self.model = model
        self.logger = logger
        self.learning_rate = cmd_args.learning_rate
        # self.optimizer = torch.optim.RMSprop(model.parameters(), lr=cmd_args.learning_rate,
        #                            weight_decay=cmd_args.learning_rate*10, momentum=cmd_args.learning_rate*100)
        if str(cmd_args.use_madam).lower() == "true":
            self.optimizer = Madam(model.parameters(), lr=cmd_args.learning_rate)
        elif str(cmd_args.use_mtadam).lower() == "true":
            self.optimizer = MTAdam(model.parameters(), lr=cmd_args.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
        self.use_mtadam = cmd_args.use_mtadam
        self.num_epochs = cmd_args.num_epochs
        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.wandb = wandb
        self.CHECKPOINT_PATH = checkpoint_path
        self.DATASET_PATH = dir_path
        self.OUTPUT_PATH = cmd_args.output_path

        self.model_name = cmd_args.model_name
        self.clip_grads = cmd_args.clip_grads
        self.with_apex = cmd_args.apex
        self.with_mip = str(cmd_args.with_mip).lower() == "true"
        self.num_classes = cmd_args.num_classes
        self.train_encoder_only = cmd_args.train_encoder_only

        # image input parameters
        self.patch_size = cmd_args.patch_size
        self.stride_depth = cmd_args.stride_depth
        self.stride_length = cmd_args.stride_length
        self.stride_width = cmd_args.stride_width
        self.samples_per_epoch = cmd_args.samples_per_epoch

        # execution configs
        self.batch_size = cmd_args.batch_size
        self.num_worker = cmd_args.num_worker

        # Losses
        self.s_ncut_loss_coeff = cmd_args.s_ncut_loss_coeff
        self.reconstr_loss_coeff = cmd_args.reconstr_loss_coeff
        self.mip_loss_coeff = cmd_args.mip_loss_coeff
        self.reg_alpha = cmd_args.reg_alpha

        # Following metrics can be used to evaluate
        self.radius = cmd_args.radius
        self.sigmaI = cmd_args.sigmaI
        self.sigmaX = cmd_args.sigmaX
        # self.soft_ncut_loss = torch.nn.DataParallel(
        #     SoftNCutsLoss(radius=self.radius, sigma_i=self.sigmaI, sigma_x=self.sigmaX))
        # self.soft_ncut_loss.cuda()
        self.soft_ncut_loss = SoftNCutsLoss(radius=self.radius, sigma_i=self.sigmaI, sigma_x=self.sigmaX)
        # self.ssim = ssim  # structural_similarity_index_measure
        # self.ssim = structural_similarity_index_measure
        # self.reconstruction_loss = torch.nn.DataParallel(
        #     ReconstructionLoss(recr_loss_model_path=cmd_args.recr_loss_model_path,
        #                        loss_type="L1"))
        # self.reconstruction_loss.cuda()
        self.reconstruction_loss = ReconstructionLoss(recr_loss_model_path=cmd_args.recr_loss_model_path,
                                                      loss_type="L1")
        self.mip_loss = FocalTverskyLoss()
        self.mip_axis = cmd_args.mip_axis
        # self.dice = Dice()
        # self.focalTverskyLoss = FocalTverskyLoss()
        # self.iou = IOU()

        # Morphological hyper-parameters
        self.init_threshold = cmd_args.init_threshold
        self.otsu_thresh_param = cmd_args.otsu_thresh_param
        self.area_opening_threshold = cmd_args.area_opening_threshold
        self.footprint_radius = cmd_args.footprint_radius

        self.save_img = str(cmd_args).lower() == "true"

        self.LOWEST_LOSS = float('inf')

        self.scaler = GradScaler()

        self.logger.info("Model Hyper Params: ")
        self.logger.info("\nLearning Rate: " + str(self.learning_rate))
        self.logger.info("\nNumber of Convolutional Blocks: " + str(cmd_args.num_conv))
        self.predictor_subject_name = cmd_args.predictor_subject_name
        if cmd_args.training_mode != "unsupervised":
            self.label_dir_path_train = self.DATASET_PATH + '/train_label/'
            self.label_dir_path_val = self.DATASET_PATH + '/validate_label/'
        else:
            self.label_dir_path_train, self.label_dir_path_val = None, None

        if cmd_args.train:  # Only if training is to be performed
            training_set = SRDataset(logger=logger, patch_size=self.patch_size,
                                     dir_path=self.DATASET_PATH + '/train/',
                                     label_dir_path=self.label_dir_path_train,
                                     output_path=self.OUTPUT_PATH,
                                     model_name=self.model_name,
                                     stride_depth=self.stride_depth, stride_length=self.stride_length,
                                     stride_width=self.stride_width, fly_under_percent=None,
                                     patch_size_us=self.patch_size, pre_interpolate=None, norm_data=False,
                                     pre_load=True,
                                     return_coords=True,
                                     threshold=self.init_threshold)
            sampler = torch.utils.data.RandomSampler(data_source=training_set, replacement=True,
                                                     num_samples=self.samples_per_epoch)
            self.train_loader = torch.utils.data.DataLoader(training_set, batch_size=self.batch_size,
                                                            num_workers=self.num_worker,
                                                            sampler=sampler, pin_memory=True)
            validation_set = Pipeline.create_tio_sub_ds(dir_path=self.DATASET_PATH + '/validate/',
                                                        label_dir_path=self.label_dir_path_val,
                                                        output_path=self.OUTPUT_PATH,
                                                        model_name=self.model_name,
                                                        patch_size=self.patch_size,
                                                        stride_length=self.stride_length,
                                                        stride_width=self.stride_width,
                                                        stride_depth=self.stride_depth,
                                                        logger=self.logger, is_validate=True,
                                                        init_threshold=self.init_threshold)
            sampler = torch.utils.data.RandomSampler(data_source=validation_set, replacement=True,
                                                     num_samples=self.samples_per_epoch)
            self.validate_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size,
                                                               shuffle=False, num_workers=self.num_worker,
                                                               pin_memory=True, sampler=sampler)

    @staticmethod
    def create_tio_sub_ds(patch_size, dir_path, stride_length, stride_width, stride_depth,
                          logger, output_path, model_name, is_validate=False,
                          get_subjects_only=False, label_dir_path=None, init_threshold=3):
        """
        Purpose: Creates 3D patches from 3D volumes using torchio and SRDataset
        :param dir_path: Path to 3D MRA volumes
        :param logger: File logger
        :param patch_size: Individual patch dimension(Only patches of equal length, width and depth are created)
        :param stride_depth: Stride Depth for patch creation
        :param stride_length: Stride Length for patch creation
        :param stride_width: Stride Width for patch creation
        :param output_path: Path to log and save thresholded images
        :param model_name: Name of the current model
        :param is_validate: Set this to true to create a tio.queue of 3D patches for validation
        :param label_dir_path: Path to GT. If provided, the MIP of the GT is computed along the 3 perceivable axes
        and the corresponding patch on the MIP of the 3D volume is returned on each lazy load of the 3D patch.
        :param init_threshold: The percentage threshold for initial histogram-based thresholding.
        :param get_subjects_only: If set, returns subjects array with 4D volumes array(channel x width x depth x height)
        """
        if is_validate:
            validation_ds = SRDataset(logger=logger, patch_size=patch_size,
                                      dir_path=dir_path,
                                      label_dir_path=label_dir_path,
                                      output_path=output_path,
                                      model_name=model_name,
                                      stride_depth=stride_depth, stride_length=stride_length,
                                      stride_width=stride_width, fly_under_percent=None,
                                      patch_size_us=patch_size, pre_interpolate=None, norm_data=False,
                                      pre_load=True,
                                      return_coords=True, threshold=init_threshold)
            overlap = np.subtract(patch_size, (stride_length, stride_width, stride_depth))
            grid_samplers = []
            for i in range(len(validation_ds)):
                grid_sampler = tio.inference.GridSampler(
                    validation_ds[i],
                    patch_size,
                    overlap,
                )
                grid_samplers.append(grid_sampler)
            return torch.utils.data.ConcatDataset(grid_samplers)
        else:
            vols = glob(dir_path + "*.nii") + glob(dir_path + "*.nii.gz")
            # labels = glob(label_dir_path + "*.nii") + glob(label_dir_path + "*.nii.gz")
            subjects = []
            for i in range(len(vols)):
                v = vols[i]
                filename = os.path.basename(v).split('.')[0]
                # l = [s for s in labels if filename in s][0]
                subject = tio.Subject(
                    img=tio.ScalarImage(v),
                    # label=tio.LabelMap(l),
                    subjectname=filename,
                )
                transforms = tio.ToCanonical(), tio.Resample(tio.ScalarImage(v))
                transform = tio.Compose(transforms)
                subject = transform(subject)
                subjects.append(subject)

            if get_subjects_only:
                return subjects

            overlap = np.subtract(patch_size, (stride_length, stride_width, stride_depth))
            grid_samplers = []
            for i in range(len(subjects)):
                grid_sampler = tio.inference.GridSampler(
                    subjects[i],
                    patch_size,
                    overlap,
                )
                grid_samplers.append(grid_sampler)
            return torch.utils.data.ConcatDataset(grid_samplers)

    @staticmethod
    def normaliser(batch):
        """
        Purpose: Normalise pixel intensities by comparing max values in the 3D patch
        :param batch: 5D array (batch_size x channel x width x depth x height)
        """
        for i in range(batch.shape[0]):
            if batch[i].max() > 0.0:
                batch[i] = batch[i] / batch[i].max()
        return batch

    def load(self, checkpoint_path=None, load_best=True, checkpoint_filename="", fold_index=""):
        """
        Purpose: Continue training from previous checkpoint or load an existing checkpoint for testing
        :param checkpoint_path: Path to the saved network state dictionary. If not specified, the path to checkpoint
        location of current directory is used.
        :param load_best: If set, uses best checkpoint from the checkpoint location. Otherwise uses last checkpoint.
        :param fold_index: Current fold number. Do not specify if debugging a specific fold.
        """
        if checkpoint_path is None:
            checkpoint_path = self.CHECKPOINT_PATH

        if checkpoint_filename == "":
            checkpoint_filename = "checkpoint"

        if self.with_apex:
            self.model, self.optimizer, self.scaler = load_model_with_amp(self.model, self.optimizer, checkpoint_path,
                                                                          batch_index="best" if load_best else "last",
                                                                          fold_index=fold_index,
                                                                          filename=checkpoint_filename)
        else:
            self.model, self.optimizer = load_model(self.model, self.optimizer, checkpoint_path,
                                                    batch_index="best" if load_best else "last",
                                                    fold_index=fold_index, filename=checkpoint_filename)

    def train(self):
        """
        Purpose: Training Pipeline including Leave-One-Out Validation.
        Performs a variety of trainings specified by command line parameters.
        """
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
            self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_soft_ncut_loss = 0
            total_reconstr_loss = 0
            total_mip_loss = 0
            total_reg_loss = 0
            total_loss = 0
            num_batches = 0

            for batch_index, patches_batch in enumerate(tqdm(self.train_loader)):

                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))
                # Clear gradients
                self.optimizer.zero_grad()

                mip_loss = torch.tensor(0.001).float().cuda()
                with autocast(enabled=self.with_apex):
                    class_preds, feature_rep, reconstructed_patch = self.model(local_batch, ops="both")
                    soft_ncut_loss = self.soft_ncut_loss(local_batch, class_preds)
                    soft_ncut_loss = self.s_ncut_loss_coeff * soft_ncut_loss.mean()

                    if self.with_mip:
                        # MIP Loss
                        feature_rep = torch.sigmoid(feature_rep)
                        # Compute MIP loss from the patch on the MIP of the 3D label and the patch prediction
                        for index, pred_patch_seg in enumerate(feature_rep):
                            # pred_patch_mip = torch.amax(pred_patch_seg, -1)
                            # Image.fromarray((pred_patch_mip.squeeze().detach().cpu().numpy() * 255)
                            # .astype('uint8'), 'L').save(os.path.join(self.OUTPUT_PATH,
                            # self.model_name + "_patch" + str(index) + "_pred_MIP.tif"))
                            # Image.fromarray((patches_batch['ground_truth_mip_patch'][index].float().squeeze()
                            # .detach().cpu().numpy() * 255).astype('uint8'), 'L').save(os.path.join(self.OUTPUT_PATH,
                            # self.model_name + "_patch" + str(index) + "_true_MIP.tif"))
                            if self.mip_axis == "multi":
                                pred_patch_mip_z = torch.amax(pred_patch_seg, -1)
                                pred_patch_mip_y = torch.amax(pred_patch_seg, 2)
                                pred_patch_mip_x = torch.amax(pred_patch_seg, 1)
                                mip_loss += (0.33 * self.mip_loss(pred_patch_mip_z,
                                             patches_batch['ground_truth_mip_z_patch'][index].float().cuda()) +
                                             0.33 * self.mip_loss(pred_patch_mip_y,
                                             patches_batch['ground_truth_mip_y_patch'][index].float().cuda()) +
                                             0.33 * self.mip_loss(pred_patch_mip_x,
                                             patches_batch['ground_truth_mip_x_patch'][index].float().cuda()))
                            else:
                                axis = -1
                                if self.mip_axis == "x":
                                    axis = 1
                                if self.mip_axis == "y":
                                    axis = 2
                                pred_patch_mip = torch.amax(pred_patch_seg, axis)
                                mip_loss += (self.mip_loss(pred_patch_mip,
                                             patches_batch[str.format('ground_truth_mip_{}_patch', self.mip_axis)]
                                             [index].float().cuda()))
                    mip_loss = self.mip_loss_coeff * (mip_loss / len(feature_rep))

                    reconstructed_patch = torch.sigmoid(reconstructed_patch)
                    reconstruction_loss = self.reconstr_loss_coeff * self.reconstruction_loss(reconstructed_patch,
                                                                                              local_batch)
                    reg_loss = self.reg_alpha * l2_regularisation_loss(self.model)
                    loss = soft_ncut_loss + mip_loss + reconstruction_loss

                    # cg = make_dot(loss, params=dict(self.model.named_parameters()))
                    # cg.render(os.path.join(self.OUTPUT_PATH, self.model_name + "cg.png"), format='png')
                    if str(self.use_mtadam).lower() == "true":
                        self.optimizer.step([soft_ncut_loss, reconstruction_loss],
                                            [self.s_ncut_loss_coeff, self.reconstr_loss_coeff])
                    else:
                        if type(loss.tolist()) is list:
                            for i in range(len(loss)):
                                if i + 1 == len(loss):  # final loss
                                    self.scaler.scale(loss[i]).backward()
                                else:
                                    self.scaler.scale(loss[i]).backward(retain_graph=True)
                            loss = torch.mean(loss)
                        else:
                            self.scaler.scale(loss).backward()
                        # self.scaler.scale(loss).backward()
                        if self.clip_grads:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                torch.cuda.empty_cache()

                training_batch_index += 1

                # Initialising the average loss metrics
                total_soft_ncut_loss += soft_ncut_loss.mean().detach().item()
                total_mip_loss += mip_loss.mean().detach().item()
                try:
                    total_reconstr_loss += reconstruction_loss.mean().detach().item()
                    total_reg_loss += reg_loss.detach().item()
                except Exception as detach_error:
                    if reconstruction_loss:
                        total_reconstr_loss += reconstruction_loss
                        total_reg_loss += reg_loss
                    else:
                        reconstruction_loss = 0
                        reg_loss = 0
                total_loss += loss.detach().item()
                if not str(self.train_encoder_only).lower() == "true":
                    reconstructed_patch.detach()
                    del reconstructed_patch
                class_preds.detach()
                del class_preds

                num_batches += 1

                self.logger.info("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + " Training..." +
                                 "\n SoftNcutLoss: " + str(soft_ncut_loss) + " ReconstructionLoss: " +
                                 str(reconstruction_loss) + "MIP-Loss: " + str(mip_loss) + " reg_loss: " + str(reg_loss)
                                 + " total_loss: " + str(loss))
                # To avoid memory errors
                torch.cuda.empty_cache()

            # Calculate the average loss per batch in one epoch
            total_soft_ncut_loss /= (num_batches + 1.0)
            total_reconstr_loss /= (num_batches + 1.0)
            total_mip_loss /= (num_batches + 1.0)
            total_reg_loss /= (num_batches + 1.0)
            total_loss /= (num_batches + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                             "\n SoftNcutLoss: " + str(total_soft_ncut_loss) +
                             " ReconstructionLoss: " + str(total_reconstr_loss) +
                             " MIP-Loss: " + str(total_mip_loss) +
                             " reg_loss: " + str(total_reg_loss) +
                             " total_loss: " + str(total_loss))
            write_epoch_summary(writer=self.writer_training, index=epoch,
                                soft_ncut_loss=total_soft_ncut_loss,
                                reconstruction_loss=total_reconstr_loss,
                                mip_loss=total_mip_loss,
                                reg_loss=total_reg_loss,
                                total_loss=total_loss)
            if self.wandb is not None:
                self.wandb.log(
                    {"SoftNcutLoss_train": total_soft_ncut_loss, "ReconstructionLoss_train": total_reconstr_loss,
                     "MIP-Loss_train": total_mip_loss, "total_reg_loss_train": total_reg_loss,
                     "total_loss_train": total_loss}, step=epoch)

            torch.cuda.empty_cache()  # to avoid memory errors
            self.validate(training_batch_index, epoch)
            torch.cuda.empty_cache()  # to avoid memory errors

        return self.model

    def validate(self, training_index, epoch):
        """
        Method to validate
        :param training_index: Epoch after which validation is performed(can be anything for test)
        :param epoch: Current training epoch
        :return:
        """
        self.logger.debug('Validating...')
        print("Validate Epoch: " + str(epoch) + " of " + str(self.num_epochs))

        total_soft_ncut_loss, total_reconstr_loss, total_mip_loss, total_loss = 0, 0, 0, 0
        no_patches = 0
        self.model.eval()
        try:
            data_loader = self.validate_loader
        except Exception as error:
            validation_set = Pipeline.create_tio_sub_ds(dir_path=self.DATASET_PATH + '/validate/',
                                                        label_dir_path=self.label_dir_path_val,
                                                        output_path=self.OUTPUT_PATH,
                                                        model_name=self.model_name,
                                                        patch_size=self.patch_size,
                                                        stride_length=self.stride_length,
                                                        stride_width=self.stride_width,
                                                        stride_depth=self.stride_depth,
                                                        logger=self.logger, is_validate=True)
            sampler = torch.utils.data.RandomSampler(data_source=validation_set, replacement=True,
                                                     num_samples=self.samples_per_epoch)
            data_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=self.num_worker,
                                                      pin_memory=True, sampler=sampler)
        writer = self.writer_validating
        with torch.no_grad():
            for index, patches_batch in enumerate(tqdm(data_loader)):
                self.logger.info("loading" + str(index))

                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                try:
                    mip_loss = torch.tensor(0.001).float().cuda()
                    with autocast(enabled=self.with_apex):
                        class_preds, feature_rep, reconstructed_patch = self.model(local_batch, ops="both")
                        soft_ncut_loss = self.soft_ncut_loss(local_batch, class_preds)
                        soft_ncut_loss = self.s_ncut_loss_coeff * soft_ncut_loss.mean()

                        if self.with_mip:
                            # MIP Loss
                            feature_rep = torch.sigmoid(feature_rep)
                            # Compute MIP loss from the patch on the MIP of the 3D label and the patch prediction
                            for idx, pred_patch_seg in enumerate(feature_rep):
                                if self.mip_axis == "multi":
                                    pred_patch_mip_z = torch.amax(pred_patch_seg, -1)
                                    pred_patch_mip_y = torch.amax(pred_patch_seg, 2)
                                    pred_patch_mip_x = torch.amax(pred_patch_seg, 1)
                                    mip_loss += (0.33 * self.mip_loss(pred_patch_mip_z,
                                                                      patches_batch['ground_truth_mip_z_patch'][
                                                                          idx].float().cuda()) +
                                                 0.33 * self.mip_loss(pred_patch_mip_y,
                                                                      patches_batch['ground_truth_mip_y_patch'][
                                                                          idx].float().cuda()) +
                                                 0.33 * self.mip_loss(pred_patch_mip_x,
                                                                      patches_batch['ground_truth_mip_x_patch'][
                                                                          idx].float().cuda()))
                                else:
                                    axis = -1
                                    if self.mip_axis == "x":
                                        axis = 1
                                    if self.mip_axis == "y":
                                        axis = 2
                                    pred_patch_mip = torch.amax(pred_patch_seg, axis)
                                    mip_loss += (self.mip_loss(pred_patch_mip,
                                                               patches_batch[str.format('ground_truth_mip_{}_patch',
                                                                                        self.mip_axis)]
                                                               [idx].float().cuda()))
                        mip_loss = self.mip_loss_coeff * (mip_loss / len(feature_rep))

                        reconstructed_patch = torch.sigmoid(reconstructed_patch)
                        reconstruction_loss = self.reconstr_loss_coeff * self.reconstruction_loss(reconstructed_patch,
                                                                                                  local_batch)

                        if not str(self.train_encoder_only).lower() == "true":
                            loss = soft_ncut_loss + mip_loss + reconstruction_loss
                        else:
                            loss = soft_ncut_loss
                        torch.cuda.empty_cache()

                except Exception as error:
                    self.logger.exception(error)

                total_soft_ncut_loss += soft_ncut_loss.detach().item()
                total_reconstr_loss += reconstruction_loss.detach().item()
                total_mip_loss += mip_loss.detach().item()
                total_loss += loss.detach().item()

                # Log validation losses
                self.logger.info("Batch_Index:" + str(index) + " Validation..." +
                                 "\n SoftNcutLoss: " + str(soft_ncut_loss) + " ReconstructionLoss: " +
                                 str(reconstruction_loss) + "MIP-Loss: " + str(mip_loss) + " total_loss: " + str(loss))
                no_patches += 1

        # Average the losses
        total_soft_ncut_loss = total_soft_ncut_loss / (no_patches + 1)
        total_reconstr_loss = total_reconstr_loss / (no_patches + 1)
        total_mip_loss = total_mip_loss / (no_patches + 1)
        total_loss = total_loss / (no_patches + 1)

        process = ' Validating'
        self.logger.info("Epoch:" + str(training_index) + process + "..." +
                         "\n SoftNcutLoss:" + str(total_soft_ncut_loss) +
                         "\n MIP-Loss:" + str(total_mip_loss) +
                         "\n ReconstructionLoss:" + str(total_reconstr_loss) +
                         "\n total_loss:" + str(total_loss))

        write_epoch_summary(writer, epoch, soft_ncut_loss=total_soft_ncut_loss,
                            reconstruction_loss=total_reconstr_loss,
                            mip_loss=total_mip_loss,
                            total_loss=total_loss)
        if self.wandb is not None:
            self.wandb.log({"SoftNcutLoss_val": total_soft_ncut_loss, "ReconstructionLoss_val": total_reconstr_loss,
                            "MIP-Loss_val": total_mip_loss, "total_loss_val": total_loss}, step=epoch)

        if self.LOWEST_LOSS > total_loss:  # Save best metric evaluation weights
            self.LOWEST_LOSS = total_loss
            self.logger.info(
                'Best metric... @ epoch:' + str(training_index) + ' Current Lowest loss:' + str(self.LOWEST_LOSS))

            if self.with_apex:
                save_model(self.CHECKPOINT_PATH, {
                    'epoch_type': 'best',
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'amp': self.scaler.state_dict()})
            else:
                save_model(self.CHECKPOINT_PATH, {
                    'epoch_type': 'best',
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'amp': None})

    def test(self, test_logger, test_subjects=None, save_results=True):
        """
        Purpose: Performs Generalization Performance Testing of the given model along with
        inference prediction on specified 3D MRA volume.
        :param test_logger: File logger
        :param save_results: If set, saves resulting segmentation as .nii.gz, the color MIP overlay of comparison of
        segmentation prediction against the ground truth.
        :param test_subjects: If specified, uses the given array of subjects(torchio subjects array) to
        prepare test loader. Otherwise creates test subjects from '/test/' and '/test_label/' folders.
        """
        test_logger.debug('Testing...')
        self.model.eval()
        result_root = os.path.join(self.OUTPUT_PATH, self.model_name, "results")
        os.makedirs(result_root, exist_ok=True)
        dir_path = self.DATASET_PATH + "/test/"
        if test_subjects is None:
            vols = glob(dir_path + "*.nii") + glob(dir_path + "*.nii.gz")
            subjects = []
            for i in range(len(vols)):
                v = vols[i]
                filename = os.path.basename(v).split('.')[0]
                # Apply same initial thresholding as the pre-trained model
                img_data = tio.ScalarImage(v)
                img_data.data = img_data.data.type(torch.float64)
                # img_data.data = torch.where((img_data.data) < 135.0, 0.0, img_data.data)
                temp_data = img_data.data.numpy().astype(np.float64)
                bins = torch.arange(temp_data.min(), temp_data.max() + 2, dtype=torch.float64)
                histogram, bin_edges = np.histogram(temp_data, int(temp_data.max() + 2))
                init_threshold = bin_edges[int(len(bins) - (1 - (self.init_threshold * 0.01)) * len(bins))]
                img_data.data = torch.where((img_data.data) <= init_threshold, 0.0, img_data.data)

                sub_dict = {
                    "img": img_data,
                    "subjectname": filename,
                    # "sampling_map": tio.Image(image_path.split('.')[0] + '_mask.nii.gz', type=tio.SAMPLING_MAP)
                }
                subject = tio.Subject(**sub_dict)
                subjects.append(subject)

        for test_subject in test_subjects:
            # test_subject = test_subjects[0]
            subjectname = test_subject['subjectname']
            overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))
            grid_sampler = tio.inference.GridSampler(
                test_subject,
                self.patch_size,
                overlap,
            )

            aggregator1 = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
            aggregator2 = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size, shuffle=False,
                                                       num_workers=self.num_worker)
            # Extract volumetric segmentation prediction by combining overlapping 3D patch inferences
            for index, patches_batch in enumerate(tqdm(patch_loader)):
                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                locations = patches_batch[tio.LOCATION]

                with autocast(enabled=self.with_apex):
                    class_preds, feature_rep = self.model(local_batch, ops="enc")
                    feature_rep = torch.sigmoid(feature_rep)
                    # reconstructed_patch = torch.sigmoid(reconstructed_patch)
                    ignore, class_assignments = torch.max(class_preds, 1, keepdim=True)
                    class_preds = class_preds.detach().type(local_batch.type())
                    # reconstructed_patch = reconstructed_patch.detach().type(local_batch.type())
                    ignore = ignore.detach()
                    class_assignments = class_assignments.detach().type(local_batch.type())
                    feature_rep = feature_rep.detach().type(local_batch.type())
                # aggregator1.add_batch(class_preds, locations)
                aggregator1.add_batch(feature_rep, locations)
                aggregator2.add_batch(class_assignments, locations)

            class_probs = aggregator1.get_output_tensor()
            class_assignments = aggregator2.get_output_tensor()
            # torch.save(class_probs, os.path.join(result_root, subjectname + "_class_probs.pth"))
            # torch.save(class_assignments, os.path.join(result_root, subjectname + "_class_assignments.pth"))

            # to avoid memory errors
            torch.cuda.empty_cache()

            # Segmentation post-processing
            # 1. Apply otsu-threshold
            thresh = threshold_otsu(class_probs.squeeze().numpy())
            class_probs = class_probs < (thresh * self.otsu_thresh_param)

            # 2. Apply morphological area-opening
            opened = area_opening(class_probs.squeeze().numpy().astype(np.uint16),
                                  area_threshold=self.area_opening_threshold)

            # 3. Apply morphological dilation
            footprint = ball(self.footprint_radius)
            dilated = dilation(opened, footprint)

            save_nifti(dilated, os.path.join(result_root, subjectname + ".nii.gz"))


            # save_nifti(class_probs.squeeze().numpy().astype(np.float32),
            #            os.path.join(result_root, subjectname + "_seg_vol.nii.gz"))
            # save_nifti(reconstructed_image.squeeze().numpy().astype(np.float32),
            #            os.path.join(result_root, subjectname + "_recr.nii.gz"))

    def predict(self, image_path, label_path, predict_logger, fold_index=""):
        """
        Purpose: Render inference on nifti 3D volume specified using pretrained WNet network
        :param image_path: Path to the nifti nii image
        :param label_path: Optionally provide the GT path
        :param predict_logger: To log exceptions
        """
        image_name = os.path.basename(image_path).split('.')[0]
        # Apply same initial thresholding as the pre-trained model
        img_data = tio.ScalarImage(image_path)
        img_data.data = img_data.data.type(torch.float64)
        # img_data.data = torch.where((img_data.data) < 135.0, 0.0, img_data.data)
        temp_data = img_data.data.numpy().astype(np.float64)
        bins = torch.arange(temp_data.min(), temp_data.max() + 2, dtype=torch.float64)
        histogram, bin_edges = np.histogram(temp_data, int(temp_data.max() + 2))
        init_threshold = bin_edges[int(len(bins) - (1 - (self.init_threshold * 0.01)) * len(bins))]
        img_data.data = torch.where(img_data.data <= init_threshold, 0.0, img_data.data)

        sub_dict = {
            "img": img_data,
            "subjectname": image_name,
            # "sampling_map": tio.Image(image_path.split('.')[0] + '_mask.nii.gz', type=tio.SAMPLING_MAP)
        }

        if bool(label_path):
            sub_dict["label"] = tio.LabelMap(label_path)

        subject = tio.Subject(**sub_dict)

        self.test(predict_logger, test_subjects=[subject], save_results=True)

    def create_vessel_diameter_mask(self, seg_path, gt_path, dim=2):
        """
        Purpose: Creates a vessel diameter mask for the given predicted segmentation
        by overlaying vessel diameters from the corresponding GT.
        :param seg_path: Path to the predicted segmentation
        :param gt_path: Path to the GT
        :param dim: The number of dimensions for analysis of vessel diameter. Use 2 for 2D and 3 for 3D.
        """
        result_root = os.path.join(self.OUTPUT_PATH, self.model_name, "results")
        os.makedirs(result_root, exist_ok=True)
        subject_name = seg_path.replace("\\", "/").split("/")[-1].split(".")[0]
        pred_seg = nib.load(seg_path).get_fdata().astype(np.uint8)
        gt_seg = nib.load(gt_path).get_fdata().astype(np.uint8)
        if dim == 2:
            pred_diameter_mask = calculate_vessel_diameter2d(pred_seg)
            gt_diameter_mask = calculate_vessel_diameter2d(gt_seg)
        elif dim == 3:
            pred_diameter_mask = calculate_vessel_diameter3d(pred_seg)
            gt_diameter_mask = calculate_vessel_diameter3d(gt_seg)
        else:
            print("input dimensionality error")
            sys.exit()
        dice_df = diameter_mask_ranking(pred_diameter_mask, gt_diameter_mask, result_root, subject_name,
                                        dim)
        if self.save_img:
            save_mask(mask=pred_diameter_mask, output_path=result_root,
                      op_filename=subject_name + "_pred_diameter_mask", dim=dim)
            save_mask(mask=gt_diameter_mask, output_path=result_root,
                      op_filename=subject_name + "_gt_diameter_mask", dim=dim)

        temp_df = pd.DataFrame.from_dict(dice_df)
        temp_df.to_csv(os.path.join(result_root, subject_name + "_diameter_df.csv"))
