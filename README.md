# DILITHIUM
DILITHIUM: Deep learning with less-to-no supervision for the segmentation of vessels in high-resolution 7T MRAs using unsupervised and weakly-supervised learning

In this study the Attention WNet proposed in the study ["Upgraded w-net with attention gates and its application in unsupervised 3d liver segmentation"](https://arxiv.org/abs/2011.10654) is adapted for performing 3D patch-based unsupervised vessel segmentation. Additionally, a [UNet MSS](https://link.springer.com/chapter/10.1007/978-3-319-67389-9_32) alternate for the encoder and decoder of the aforementioned WNet is proposed. The MIP-based weak supervision approach is also proposed in this study and the implementation of the proposed methods is presented in this source code.

##Setup Instructions

All experiments are performed using Tesla V100-SXM2
GPU via NVLink provided by OVGU GPU18 HPC8. The GPU
node is equipped with a GPU memory of 32GB and a maximum
of 10 CPU threads with a memory of 6GB per CPU. The cluster
runs Ubuntu 20.04.3 LTS with CUDA 11.4. The HPC consists
of 8 such GPU nodes with a limitation of one node per user and
each GPU process is allowed to run for a maximum period of 6
days. Therefore, the environment for the experiments is tailored
to suit the memory and time constraints.

### Dependencies
 - python ^3.8
 - pytorch ^1.11.0
 - torchio ^0.18
 - nibabel ^3.2.2
 - numpy ^1.22.3
 - pandas ^1.4.1
 - scipy ^1.8.0
 - scikit-image ^0.19.2

## Run Instructions

The implementation covers a wide range of training, testing and inference code. Please follow the instructions below for each specific method and feature.

### Generic Hyper-parameters

The values for each the generic hyper-parameters specified below is curated for the GPU on which the experiments were run.
```code
-batch_size 16 -patch_size 32 -learning_rate 0.0001 -num_epochs 50 -stride_depth 16 -stride_width 28 -stride_length 28 -num_worker 8 -init_threshold 3
```

*Use stride diemnsions of 2x2x2 for inference

### Train Unsupervised WNet
```code
python main.py -model 2 -model_name <name_of_the_current_run> -dataset_path <path_to_dataset_folder> -output_path <path_to_output_folder> -training_mode "unsupervised" -train True -s_ncut_loss_coeff 0.1 -reconstr_loss_coeff 1.0 -radius 4 -num_classes 5 ...<generic_hyper_parameters>
```

Select relevant model by selecting -model as 1 (for UNet), 2 (for Attention UNet) or 3 (for UNet MSS) for the encoder and decoder of WNet.
Dataset folder can be organized as illustrated below:<br/>
If cross validation is to be performed, please create the dataset folder consisting of
train and train_label folders along with test and test_label folders
(include label folder for weakly-supervised / MIP methods)
e.g.,<br/>
/sample_dataset<br/>
/train<br/>
/train_label<br/>
/test<br/>
/test_label<br/>
Otherwise prepare the dataset folder consisting of
train, train_label, validate, validate_label, test and test_label folders
(include label folder for weakly-supervised / MIP methods)
e.g.,<br/>
/sample_dataset<br/>
/train<br/>
/train_label<br/>
/validate<br/>
/validate_label<br/>
/test<br/>
/test_label<br/>
Each folder must contain at least one 3D MRA volume in nifti .nii or nii.gz formats

### Train Weakly-supervised WNet
```code
python main.py -model 2 -model_name <name_of_the_current_run> -dataset_path <path_to_dataset_folder> -output_path <path_to_output_folder> -training_mode "weakly-supervised" -train True -s_ncut_loss_coeff 0.0 -reconstr_loss_coeff 1.0 -mip_loss_coeff 0.1 -num_classes 5 -with_mip True -mip_axis "z" ...<generic_hyper_parameters>
```

Select the MIP axis by selecting -mip_axis as "x" (for x-axis), "y" (for y-axis), "z" (for z-axis) or "multi" (for multi-axial MIP loss).

### Generate inference on 3D nifti volume
The inference can be drawn on a specific 3D volume or a set of 3D volumes in a directory.<br/>
The following inference hyper-parameters are tuned for the experiments:
```code
-stride_depth 2 -stride_width 2 -stride_length 2 -otsu_thresh_param 0.1 -area_opening_threshold 32 -footprint_radius 1 
```
Set -otsu_thresh_param as 0.3 (for Weakly-supervised Att. MIP and mMIP methods) or 0.7 (otherwise). Set -area_opening_threshold as 8. 
*These params can be further tuned.

For a single 3D volume:
```code
python main.py -model 2 -pre_train True -load_path <path_to_checkpoint.pth> -checkpoint_filename <if_other_than_'checkpointbest.pth'> -dataset_path <path_to_folder_containing_nii> -output_path <path_to_output_folder> -predictor_path <path_to_nii.gz> -predict True ...<inference_hyper_parameters>
```
Select -model depending on the pre-trained model.

For a set of 3D volumes:
```code
python main.py -model 2 -pre_train True -load_path <path_to_checkpoint.pth> -checkpoint_filename <if_other_than_'checkpointbest.pth'> -dataset_path <path_to_folder_containing_nii> -output_path <path_to_output_folder> -test True ...<inference_hyper_parameters>
```
*The inference is a time-consuming process as it involves overlapping patches of size 32x32x32 over the entire 3D volume of size 480x640x164.

### Creating Vessel Diameter Mask
The implementation includes a novel evaluation method where the quality of the segmented vasculature of different size can be compared between the predicted segmentation and the ground truth.
This is achieved by computing a vessel diameter mask. This can be computed using:
```code
python main.py -create_vessel_diameter_mask True -seg_path <path_to_predicted_segmentation_nii.gz> -gt_path <path_to_gt_segmentation_nii.gz> -output_path <path_to_output_folder> -dim <2 for 2D or 3 for 3D> 
```
The method returns 2D mask for -dim 2 and 3D mask for -dim 3. It also returns a DF.csv containing the dice scores evaluated against different vessel sizes (diameters).

### Cross Validation
The trainings can be performed with k-fold cross validation by specifying additional parameters to the aforementioned training cmd.
```code
-cross_validate True -k_folds 5
```
Set the number of folds using -k_folds. Please organize the dataset folder as mentioned above.

### Using WandB for logging
The implementation is integrated with the wandb api. Use the following parameters to configure the wandb api.
```code
-wandb True -wandb_project <wandb_project_name> -wandb_entity <wandb_entity_name> -wandb_api_key <wandb_api_key>
```
The wandb config properties can be found at https://wandb.ai/authorize after creating the project.

### **args

| arg | description | default |
| :----- | :------------- | :---------: |
| -model | 1{U-Net}; 2{U-Net_Deepsup}; 3{Attention-U-Net} | 2 |
| -model_name | Name of the run | "Model_v1" |
| -dataset_path | Path to folder containing dataset | "" |
| -output_path | Folder path to store output | "" |
| -training_mode | Training Mode: 'unsupervised' or 'weakly-supervised' | "unsupervised" |
| -train | To train the model | False |
| -test | To test the model | False |
| -predict | To predict a segmentation output of the model and to get a diff between label and output | False |
| -eval | Set this to true for running in production and just for evaluation | False |
| -pre_train | Set this to true to load pre-trained model | False |
| -create_vessel_diameter_mask | To create a vessel diameter mask for input predicted segmentation against GT | False |
| -predictor_path | Path to the input image to predict an output, ex:/home/test/ww25.nii | "" |
| -predictor_label_path | Path to the label image to find the diff between label an output | "" |
| -seg_path | Path to the predicted segmentation, ex:/home/test/ww25.nii | "" |
| -gt_path | Path to the ground truth segmentation, ex:/home/test/ww25.nii | "" |
| -dim | number of dimensions for creating diameter mask. Use 2 for 2D mask and 3 for 3D mask. | 2 |
| -save_img | Set this to true to save images in the /results folder | True |
| -recr_loss_model_path | Path to weights '.pth' file for unet3D model used to compute reconstruction loss | "" |
| -load_path | Path to checkpoint of existing model to load, ex:/home/model/checkpoint | "" |
| -checkpoint_filename | Provide filename of the checkpoint if different from 'checkpoint' | "" |
| -load_best | Specifiy whether to load the best checkpoiont or the last. Also to be used if Train and Test both are true. | True |
| -clip_grads | To use deformation for training | True |
| -apex | To use half precision on model weights | True |
| -cross_validate | To train with k-fold cross validation | False |
| -num_conv | Number of convolutional layers in each convolutional block of the encoder and decoder | 2 |
| -num_classes | Number of classes K, estimated and optimized by the WNet | 5 |
| -batch_size | Batch size for training | 15 |
| -num_epochs | Number of epochs for training | 50 |
| -learning_rate | Learning rate | 0.0001 |
| -patch_size | Patch size of the input volume | 32 |
| -stride_depth, stride_width, stride_length | Strides for dividing the input volume into patches in depth dimension (To be used during validation and inference) | 16, 28, 28 |
| -num_worker | Number of worker threads | 8 |
| -s_ncut_loss_coeff | loss coefficient for soft ncut loss | 0.1 |
| -reconstr_loss_coeff | loss coefficient for reconstruction loss | 1.0 |
| -mip_loss_coeff | loss coefficient for maximum intensity projection loss | 0.1 |
| -mip_axis | Set projection axis. Default is z-axis. use axis in \[x, y, z\] or 'multi' | z |
| -reg_alpha | loss coefficient for regularisation loss | 0.0 |
| -predictor_subject_name | subject name of the predictor image | "" |
| -radius | radius of the voxel | 4 |
| -sigmaX | weight for spation correlation in soft_ncut_loss calculation | 4 |
| -sigmaI | weight for pixel intensity correlation in soft_ncut_loss calculation | 10 |
| -init_threshold | Initial histogram threshold (in %) | 3 |
| -otsu_thresh_param | parameter for otsu thresholding. Use 0.1 for unsupervised and 0.5 for weakly-supervised. | 0.1 |
| -area_opening_threshold | parameter for morphological area-opening. Use 32 for unsupervised and 8 for weakly-supervised. | 32 |
| -footprint_radius | radius of structural footprint for morphological dilation | 1 |
| -fold_index | Fold Index for loading the model trained with cross validation | "" |
| -k_folds | Number of folds for K-fold cross validation | 5 |
| -wandb | Set this to true to include wandb logging | False |
| -wandb_project | Set this to wandb project name e.g., 'DS6_VesselSeg2' | "" |
| -wandb_entity | Set this to wandb project name e.g., 'ds6_vessel_seg2' | "" |
| -wandb_api_key | API Key to login that can be found at https://wandb.ai/authorize | "" |
| -train_encoder_only | Set this to True to train only the encoder | False |
| -with_mip | Set this to true to train with mip loss | False |
| -use_madam | Set this to true to use madam optimizer | False |
| -use_FTL | Set this to true to use FocalTverskyLoss for similarity loss | False |
| -use_mtadam | Set this to true to use mTadam optimizer | False |