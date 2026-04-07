#!/usr/bin/env bash
# finetune_weaklysup.sh
# Fine-tune the weakly-supervised DILITHIUM models on a new dataset,
# starting from pretrained weights on HuggingFace Hub.
#
# Usage:
#   1. Set DATASET_PATH – folder with train/, train_label/, validate/, validate_label/, test/ sub-folders.
#   2. Set OUTPUT_PATH – where checkpoints, logs and results will be saved.
#   3. (Optional) adjust the hyper-parameters below.
#   4. Run:  bash finetune_weaklysup.sh
#      or:   uv run bash finetune_weaklysup.sh
#
# Dataset structure required:
#   $DATASET_PATH/
#     train/            ← 3D MRA volumes (.nii / .nii.gz)
#     train_label/      ← MIP projection labels (.nii / .nii.gz)
#     validate/         ← 3D MRA volumes (.nii / .nii.gz)
#     validate_label/   ← MIP projection labels (.nii / .nii.gz)
#     test/             ← 3D MRA volumes (.nii / .nii.gz)

set -euo pipefail

DATASET_PATH="/path/to/your/dataset"
OUTPUT_PATH="/path/to/your/output"

# Fine-tuning hyper-parameters
NUM_EPOCHS=20
LEARNING_RATE=0.00005        # lower than original (0.0001) to preserve pretrained features
BATCH_SIZE=16
PATCH_SIZE=32
NUM_CLASSES=5
NUM_WORKER=8
STRIDE_DEPTH=16
STRIDE_WIDTH=28
STRIDE_LENGTH=28

COMMON_TRAIN="-training_mode weakly-supervised
              -train True
              -test True
              -with_mip True
              -num_epochs ${NUM_EPOCHS}
              -learning_rate ${LEARNING_RATE}
              -batch_size ${BATCH_SIZE}
              -patch_size ${PATCH_SIZE}
              -num_classes ${NUM_CLASSES}
              -num_worker ${NUM_WORKER}
              -stride_depth ${STRIDE_DEPTH}
              -stride_width ${STRIDE_WIDTH}
              -stride_length ${STRIDE_LENGTH}
              -s_ncut_loss_coeff 0.0
              -reconstr_loss_coeff 1.0
              -mip_loss_coeff 0.1
              -radius 4
              -init_threshold 3
              -otsu_thresh_param 0.3
              -area_opening_threshold 8
              -footprint_radius 1
              -dataset_path ${DATASET_PATH}
              -output_path  ${OUTPUT_PATH}"

# ---------------------------------------------------------------------------
# 1. Fine-tune WeaklySup WNetMSS3D with multi-MIP
# ---------------------------------------------------------------------------
echo "=== [1/4] Fine-tuning WeaklySup WNetMSS3D mMIP ==="
uv run python main.py \
    -model 2 \
    -model_name "Finetune_WeaklySup_WNetMSS3D_mMIP" \
    -mip_axis "multi" \
    -hf_model "soumickmj/DILITHIUM_WeaklySup_WNetMSS3D_mMIP" \
    ${COMMON_TRAIN}

# ---------------------------------------------------------------------------
# 2. Fine-tune WeaklySup AttWNet3D with multi-MIP
# ---------------------------------------------------------------------------
echo "=== [2/4] Fine-tuning WeaklySup AttWNet3D mMIP ==="
uv run python main.py \
    -model 3 \
    -model_name "Finetune_WeaklySup_AttWNet3D_mMIP" \
    -mip_axis "multi" \
    -hf_model "soumickmj/DILITHIUM_WeaklySup_AttWNet3D_mMIP" \
    ${COMMON_TRAIN}

# ---------------------------------------------------------------------------
# 3. Fine-tune WeaklySup WNetMSS3D with single-axis MIP (z)
# ---------------------------------------------------------------------------
echo "=== [3/4] Fine-tuning WeaklySup WNetMSS3D MIP ==="
uv run python main.py \
    -model 2 \
    -model_name "Finetune_WeaklySup_WNetMSS3D_MIP" \
    -mip_axis "z" \
    -hf_model "soumickmj/DILITHIUM_WeaklySup_WNetMSS3D_MIP" \
    ${COMMON_TRAIN}

# ---------------------------------------------------------------------------
# 4. Fine-tune WeaklySup AttWNet3D with single-axis MIP (z)
# ---------------------------------------------------------------------------
echo "=== [4/4] Fine-tuning WeaklySup AttWNet3D MIP ==="
uv run python main.py \
    -model 3 \
    -model_name "Finetune_WeaklySup_AttWNet3D_MIP" \
    -mip_axis "z" \
    -hf_model "soumickmj/DILITHIUM_WeaklySup_AttWNet3D_MIP" \
    ${COMMON_TRAIN}

echo "=== Fine-tuning complete. Checkpoints saved to: ${OUTPUT_PATH} ==="
