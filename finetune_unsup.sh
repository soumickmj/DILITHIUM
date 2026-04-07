#!/usr/bin/env bash
# finetune_unsup.sh
# Fine-tune the unsupervised DILITHIUM models (WNetMSS3D and AttWNet3D) on a new dataset,
# starting from pretrained weights on HuggingFace Hub.
#
# Usage:
#   1. Set DATASET_PATH – folder with train/, validate/, test/ sub-folders
#      (no label folders needed for unsupervised fine-tuning).
#   2. Set OUTPUT_PATH – where checkpoints, logs and results will be saved.
#   3. (Optional) adjust the hyper-parameters below.
#   4. Run:  bash finetune_unsup.sh
#      or:   uv run bash finetune_unsup.sh
#
# Dataset structure required:
#   $DATASET_PATH/
#     train/      ← 3D MRA volumes (.nii / .nii.gz)
#     validate/   ← 3D MRA volumes (.nii / .nii.gz)
#     test/       ← 3D MRA volumes (.nii / .nii.gz)

set -euo pipefail

DATASET_PATH="/path/to/your/dataset"
OUTPUT_PATH="/path/to/your/output"

# Fine-tuning hyper-parameters — reduce epochs/lr for a new domain
NUM_EPOCHS=20
LEARNING_RATE=0.00005        # lower than original (0.0001) to preserve pretrained features
BATCH_SIZE=16
PATCH_SIZE=32
NUM_CLASSES=5
NUM_WORKER=8
STRIDE_DEPTH=16
STRIDE_WIDTH=28
STRIDE_LENGTH=28

COMMON_TRAIN="-training_mode unsupervised
              -train True
              -test True
              -num_epochs ${NUM_EPOCHS}
              -learning_rate ${LEARNING_RATE}
              -batch_size ${BATCH_SIZE}
              -patch_size ${PATCH_SIZE}
              -num_classes ${NUM_CLASSES}
              -num_worker ${NUM_WORKER}
              -stride_depth ${STRIDE_DEPTH}
              -stride_width ${STRIDE_WIDTH}
              -stride_length ${STRIDE_LENGTH}
              -s_ncut_loss_coeff 0.1
              -reconstr_loss_coeff 1.0
              -radius 4
              -init_threshold 3
              -otsu_thresh_param 0.7
              -area_opening_threshold 32
              -footprint_radius 1
              -dataset_path ${DATASET_PATH}
              -output_path  ${OUTPUT_PATH}"

# ---------------------------------------------------------------------------
# 1. Fine-tune Unsupervised WNetMSS3D
# ---------------------------------------------------------------------------
echo "=== [1/2] Fine-tuning UnSup WNetMSS3D ==="
uv run python main.py \
    -model 2 \
    -model_name "Finetune_UnSup_WNetMSS3D" \
    -hf_model "soumickmj/DILITHIUM_UnSup_WNetMSS3D" \
    ${COMMON_TRAIN}

# ---------------------------------------------------------------------------
# 2. Fine-tune Unsupervised AttWNet3D
# ---------------------------------------------------------------------------
echo "=== [2/2] Fine-tuning UnSup AttWNet3D ==="
uv run python main.py \
    -model 3 \
    -model_name "Finetune_UnSup_AttWNet3D" \
    -hf_model "soumickmj/DILITHIUM_UnSup_AttWNet3D" \
    ${COMMON_TRAIN}

echo "=== Fine-tuning complete. Checkpoints saved to: ${OUTPUT_PATH} ==="
