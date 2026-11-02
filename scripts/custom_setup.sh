#!/usr/bin/env bash
# Setup script for fine-tuning FastWAM on custom humanoid data.
# Run once before training. Safe to re-run.
set -euo pipefail

# Usage: bash scripts/custom_setup.sh /path/to/your_dataset
DATASET_DIR="${1:?usage: bash scripts/custom_setup.sh /path/to/your_dataset}"
DATASET_DIR="${DATASET_DIR%/}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== [1/3] Setting up checkpoint directory ==="
mkdir -p "${REPO_ROOT}/checkpoints"
export DIFFSYNTH_MODEL_BASE_PATH="${REPO_ROOT}/checkpoints"
echo "DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}"

echo ""
echo "=== [2/3] Generating ActionDiT backbone ==="
echo "  NOTE: Will auto-download Wan2.2-TI2V-5B from ModelScope on first run (~21 GB total)."
echo "        Set DIFFSYNTH_DOWNLOAD_SOURCE=huggingface to use HuggingFace instead."
ACTIONDIT_OUT="${REPO_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
if [ -f "${ACTIONDIT_OUT}" ]; then
    echo "  ActionDiT backbone already exists, skipping."
else
    cd "${REPO_ROOT}"
    python scripts/preprocess_action_dit_backbone.py \
        --model-config configs/model/fastwam.yaml \
        --output "${ACTIONDIT_OUT}" \
        --device cuda \
        --dtype bfloat16
fi

echo ""
echo "=== [3/3] Precomputing T5 text embeddings ==="
echo "  NOTE: Tokenizer and text encoder will be auto-downloaded if not present."
cd "${REPO_ROOT}"
python scripts/precompute_text_embeds.py \
    --config-name train \
    data=custom_1cam \
    model=fastwam \
    "data.train.dataset_dirs=[${DATASET_DIR}]" \
    "data.train.text_embedding_cache_dir=${DATASET_DIR}/cache_t5" \
    overwrite=false

echo ""
echo "=== Setup complete. ==="
echo "Run training with:"
echo "  bash scripts/train_zero1.sh <NUM_GPUS> task=custom_1cam_1e-4 \\"
echo "    \"data.train.dataset_dirs=[${DATASET_DIR}]\" data.train.text_embedding_cache_dir=${DATASET_DIR}/cache_t5"
