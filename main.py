#!/usr/bin/env python
"""

"""

import argparse
import random

# import nibabel as nib
# from glob import glob
# import os
import numpy as np
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from pipeline import Pipeline
from cross_validation_pipeline import CrossValidationPipeline
from utils.logger import Logger
from utils.model_manager import get_model

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(2022)
np.random.seed(2022)
random.seed(2022)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-model",
                        type=int,
                        default=3,
                        help="1{U-Net}; \n"
                             "2{U-Net_Deepsup}; \n"
                             "3{Attention-U-Net};")
    parser.add_argument("-model_name",
                        default="Model_v1",
                        help="Name of the model")
    parser.add_argument("-dataset_path",
                        default="",
                        help="Path to folder containing dataset.")
    parser.add_argument("-output_path",
                        default="",
                        help="Folder path to store output "
                             "Example: /home/output/")
    parser.add_argument("-training_mode",
                        default="unsupervised",
                        help="Training Mode: 'unsupervised' or 'supervised'")
    parser.add_argument('-train',
                        default=True,
                        help="To train the model")
    parser.add_argument('-test',
                        default=False,
                        help="To test the model")
    parser.add_argument('-predict',
                        default=False,
                        help="To predict a segmentation output of the model and to get a diff between label and output")
    parser.add_argument('-testing_validation',
                        default=False,
                        help="To train the model")
    parser.add_argument('-predictor_path',
                        default="",
                        help="Path to the input image to predict an output, ex:/home/test/ww25.nii ")
    parser.add_argument('-predictor_label_path',
                        default="",
                        help="Path to the label image to find the diff between label an output"
                             ", ex:/home/test/ww25_label.nii ")
    parser.add_argument('-recr_loss_model_path',
                        default="",
                        help="Path to weights '.pth' file for unet3D model used to compute reconstruction loss")
    parser.add_argument('-load_path',
                        default="",
                        help="Path to checkpoint of existing model to load, ex:/home/model/checkpoint")
    parser.add_argument('-load_best',
                        default=True,
                        help="Specifiy whether to load the best checkpoiont or the last. "
                             "Also to be used if Train and Test both are true.")
    parser.add_argument('-clip_grads',
                        default=True,
                        action="store_true",
                        help="To use deformation for training")
    parser.add_argument('-apex',
                        default=True,
                        help="To use half precision on model weights.")
    parser.add_argument('-cross_validate',
                        default=False,
                        help="To train with k-fold cross validation")
    parser.add_argument("-num_conv",
                        type=int,
                        default=3,
                        help="Batch size for training")
    parser.add_argument("-num_classes",
                        type=int,
                        default=6,
                        help="Batch size for training")
    parser.add_argument("-batch_size",
                        type=int,
                        default=15,
                        help="Batch size for training")
    parser.add_argument("-num_epochs",
                        type=int,
                        default=20,
                        help="Number of epochs for training")
    parser.add_argument("-learning_rate",
                        type=float,
                        default=0.0001,
                        help="Learning rate")
    parser.add_argument("-patch_size",
                        type=int,
                        default=32,
                        help="Patch size of the input volume")
    parser.add_argument("-stride_depth",
                        type=int,
                        default=16,
                        help="Strides for dividing the input volume into patches in "
                             "depth dimension (To be used during validation and inference)")
    parser.add_argument("-stride_width",
                        type=int,
                        default=28,
                        help="Strides for dividing the input volume into patches in "
                             "width dimension (To be used during validation and inference)")
    parser.add_argument("-stride_length",
                        type=int,
                        default=28,
                        help="Strides for dividing the input volume into patches in "
                             "length dimension (To be used during validation and inference)")
    parser.add_argument("-samples_per_epoch",
                        type=int,
                        default=8000,
                        help="Number of samples per epoch")
    parser.add_argument("-num_worker",
                        type=int,
                        default=8,
                        help="Number of worker threads")
    parser.add_argument("-s_ncut_loss_coeff",
                        type=float,
                        default=1.0,
                        help="loss coefficient for soft ncut loss")
    parser.add_argument("-reconstr_loss_coeff",
                        type=float,
                        default=1.0,
                        help="loss coefficient for reconstruction loss")
    parser.add_argument("-mip_loss_coeff",
                        type=float,
                        default=0.3,
                        help="loss coefficient for maximum intensity projection loss")
    parser.add_argument("-mip_axis",
                        type=str,
                        default="z",
                        help="Set projection axis. Default is z-axis. use axis in [x, y, z] or 'multi'")
    parser.add_argument("-reg_alpha",
                        type=float,
                        default=0.001,
                        help="loss coefficient for regularisation loss")
    parser.add_argument("-predictor_subject_name",
                        default="test_subject",
                        help="subject name of the predictor image")
    parser.add_argument("-radius",
                        type=int,
                        default=4,
                        help="radius of the voxel")
    parser.add_argument("-sigmaI",
                        type=int,
                        default=10,
                        help="SigmaI")
    parser.add_argument("-sigmaX",
                        type=int,
                        default=4,
                        help="SigmaX")
    parser.add_argument("-init_thresh",
                        type=int,
                        default=3,
                        help="Initial histogram threshold (in %)")
    parser.add_argument("-fold_index",
                        type=str,
                        default="",
                        help="fold index")
    parser.add_argument("-k_folds",
                        type=int,
                        default=5,
                        help="number of folds")
    parser.add_argument("-wandb",
                        default=True,
                        help="Set this to true to include wandb logging")
    parser.add_argument("-train_encoder_only",
                        default=False,
                        help="Set this to true to include wandb logging")
    parser.add_argument("-with_mip",
                        default=False,
                        help="Set this to true to train with mip loss")
    parser.add_argument("-use_madam",
                        default=False,
                        help="Set this to true to use madam optimizer")
    parser.add_argument("-use_FTL",
                        default=False,
                        help="Set this to true to use FocalTverskyLoss")
    parser.add_argument("-use_mtadam",
                        default=False,
                        help="Set this to true to use mTadam optimizer")
    # parser.add_argument("-create_brain_mask",
    #                     type=bool,
    #                     default=False,
    #                     help="Create binary brain mask for input volumes")

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    DATASET_FOLDER = args.dataset_path
    OUTPUT_PATH = args.output_path

    LOAD_PATH = args.load_path
    CHECKPOINT_PATH = OUTPUT_PATH + "/" + MODEL_NAME + '/checkpoint/'
    TENSORBOARD_PATH_TRAINING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_training/'
    TENSORBOARD_PATH_VALIDATION = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_validation/'
    TENSORBOARD_PATH_TESTING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_testing/'

    LOGGER_PATH = OUTPUT_PATH + "/" + MODEL_NAME + '.log'

    logger = Logger(MODEL_NAME, LOGGER_PATH).get_logger()
    test_logger = Logger(MODEL_NAME + '_test', LOGGER_PATH).get_logger()
    wandb = None
    if str(args.wandb).lower() == "true":
        import wandb

        wandb.init(project="w-net-3d", entity="chethanmysuru", notes=args.model_name)
        wandb.config = {
            "learning_rate": args.learning_rate,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "num_classes": args.num_classes,
            "samples_per_epoch": args.samples_per_epoch,
            "s_ncut_loss_coeff": args.s_ncut_loss_coeff,
            "reconstr_loss_coeff": args.reconstr_loss_coeff,
            "radius": args.radius,
            "SigmaI": args.sigmaI,
            "sigmaX": args.sigmaX
        }

    # if args.create_brain_mask:
    #     vols = glob(os.path.join(DATASET_FOLDER, "validate/") + "*.nii") + \
    #            glob(os.path.join(DATASET_FOLDER, "validate/") + "*.nii.gz")
    #     for vol in vols:
    #         filename = os.path.basename(vol).split('.')[0]
    #         if "_mask" in vol:
    #             continue
    #         print("Creating mask for {}".format(filename))
    #         vol = nib.load(vol)
    #         affine, vol_data = vol.affine, vol.get_fdata()
    #         binary_mask = np.where(vol_data > 0, 255, 0).astype(np.uint16)
    #         binary_mask = nib.Nifti1Image(binary_mask, affine=affine)
    #         nib.save(binary_mask, os.path.join(DATASET_FOLDER, "validate/") + filename + "_mask.nii.gz")
    #
    #     exit()

    # models
    model = torch.nn.DataParallel(get_model(model_no=args.model, output_ch=args.num_classes))
    model.cuda()

    writer_training = SummaryWriter(TENSORBOARD_PATH_TRAINING)
    writer_validating = SummaryWriter(TENSORBOARD_PATH_VALIDATION)

    if str(args.cross_validate).lower() == "true":
        pipeline = CrossValidationPipeline(cmd_args=args, model=model, logger=logger,
                                           dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH,
                                           writer_training=writer_training, writer_validating=writer_validating,
                                           wandb=wandb)
    else:
        pipeline = Pipeline(cmd_args=args, model=model, logger=logger,
                            dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH,
                            writer_training=writer_training, writer_validating=writer_validating, wandb=wandb)

    # loading existing checkpoint if supplied
    if bool(LOAD_PATH):
        torch.cuda.empty_cache()
        pipeline.load(checkpoint_path=LOAD_PATH, load_best=args.load_best, fold_index=args.fold_index)
        torch.cuda.empty_cache()

    try:

        if args.train:
            pipeline.train()
            torch.cuda.empty_cache()  # to avoid memory errors

        if args.test:
            if args.load_best:
                pipeline.load(load_best=True, fold_index=args.fold_index)
            pipeline.test(test_logger=test_logger)
            torch.cuda.empty_cache()  # to avoid memory errors

        if args.predict:
            if args.load_best:
                pipeline.load(load_best=True, fold_index=args.fold_index)
            pipeline.predict(predict_logger=test_logger, image_path=args.predictor_path,
                             label_path=args.predictor_label_path, fold_index=args.fold_index)
            # class_preds = torch.load(args.predictor_path)
            # pipeline.extract_segmentation(class_preds)
        if args.testing_validation:
            pipeline.validate(0, 0)
            torch.cuda.empty_cache()  # to avoid memory errors

    except Exception as error:
        print(error)
        logger.exception(error)

    writer_training.close()
    writer_validating.close()
