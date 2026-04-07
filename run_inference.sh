#!/usr/bin/env bash
# run_inference.sh
# Run inference on the test set for all 6 pretrained DILITHIUM models stored on HuggingFace Hub.
#
# Usage:
#   1. Set DATASET_PATH to the folder containing your test data (must have a "test/" sub-folder).
#   2. Set OUTPUT_PATH to the folder where results should be written.
#   3. Run:  bash run_inference.sh
#      or:   uv run bash run_inference.sh
#
# Dataset structure required:
#   $DATASET_PATH/
#     test/          ← 3D MRA volumes in .nii / .nii.gz format
#     test_label/    ← corresponding GT labels (required for weakly-supervised runs)

set -euo pipefail

DATASET_PATH="/path/to/your/dataset"
OUTPUT_PATH="/path/to/your/output"

# Common inference hyper-parameters (tuned as per the paper)
COMMON="-batch_size 16 -patch_size 32 -num_worker 8
        -stride_depth 2 -stride_width 2 -stride_length 2
        -footprint_radius 1 -area_opening_threshold 8
        -num_classes 5
        -dataset_path ${DATASET_PATH}
        -output_path  ${OUTPUT_PATH}
        -test True"

# ---------------------------------------------------------------------------
# 1. Unsupervised  –  WNetMSS3D  (model 2)
# ---------------------------------------------------------------------------
echo "=== [1/6] UnSup WNetMSS3D ==="
uv run python main.py \
    -model 2 \
    -model_name "UnSup_WNetMSS3D" \
    -training_mode "unsupervised" \
    -hf_model "soumickmj/DILITHIUM_UnSup_WNetMSS3D" \
    -otsu_thresh_param 0.7 \
    ${COMMON}

# ---------------------------------------------------------------------------
# 2. Unsupervised  –  AttWNet3D  (model 3)
# ---------------------------------------------------------------------------
echo "=== [2/6] UnSup AttWNet3D ==="
uv run python main.py \
    -model 3 \
    -model_name "UnSup_AttWNet3D" \
    -training_mode "unsupervised" \
    -hf_model "soumickmj/DILITHIUM_UnSup_AttWNet3D" \
    -otsu_thresh_param 0.7 \
    ${COMMON}

# ---------------------------------------------------------------------------
# 3. Weakly-supervised  –  WNetMSS3D  –  multi-MIP  (model 2)
# ---------------------------------------------------------------------------
echo "=== [3/6] WeaklySup WNetMSS3D mMIP ==="
uv run python main.py \
    -model 2 \
    -model_name "WeaklySup_WNetMSS3D_mMIP" \
    -training_mode "weakly-supervised" \
    -with_mip True \
    -mip_axis "multi" \
    -hf_model "soumickmj/DILITHIUM_WeaklySup_WNetMSS3D_mMIP" \
    -otsu_thresh_param 0.3 \
    ${COMMON}

# ---------------------------------------------------------------------------
# 4. Weakly-supervised  –  AttWNet3D  –  multi-MIP  (model 3)
# ---------------------------------------------------------------------------
echo "=== [4/6] WeaklySup AttWNet3D mMIP ==="
uv run python main.py \
    -model 3 \
    -model_name "WeaklySup_AttWNet3D_mMIP" \
    -training_mode "weakly-supervised" \
    -with_mip True \
    -mip_axis "multi" \
    -hf_model "soumickmj/DILITHIUM_WeaklySup_AttWNet3D_mMIP" \
    -otsu_thresh_param 0.3 \
    ${COMMON}

# ---------------------------------------------------------------------------
# 5. Weakly-supervised  –  WNetMSS3D  –  single-axis MIP  (model 2)
# ---------------------------------------------------------------------------
echo "=== [5/6] WeaklySup WNetMSS3D MIP ==="
uv run python main.py \
    -model 2 \
    -model_name "WeaklySup_WNetMSS3D_MIP" \
    -training_mode "weakly-supervised" \
    -with_mip True \
    -mip_axis "z" \
    -hf_model "soumickmj/DILITHIUM_WeaklySup_WNetMSS3D_MIP" \
    -otsu_thresh_param 0.3 \
    ${COMMON}

# ---------------------------------------------------------------------------
# 6. Weakly-supervised  –  AttWNet3D  –  single-axis MIP  (model 3)
# ---------------------------------------------------------------------------
echo "=== [6/6] WeaklySup AttWNet3D MIP ==="
uv run python main.py \
    -model 3 \
    -model_name "WeaklySup_AttWNet3D_MIP" \
    -training_mode "weakly-supervised" \
    -with_mip True \
    -mip_axis "z" \
    -hf_model "soumickmj/DILITHIUM_WeaklySup_AttWNet3D_MIP" \
    -otsu_thresh_param 0.3 \
    ${COMMON}

echo "=== All inference runs complete. Results saved to: ${OUTPUT_PATH} ==="
