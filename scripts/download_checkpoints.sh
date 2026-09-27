#!/usr/bin/env bash
# ==============================================================================
# EfficientSAM3 Checkpoint Downloader
# Downloads distilled checkpoints (TinyViT, EfficientViT, RepViT) from Hugging Face
# ==============================================================================

set -euo pipefail

DEST_DIR="${1:-checkpoints/efficientsam3_ft}"
mkdir -p "$DEST_DIR"

BASE_URL="https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/efficientsam3_ft"

MODELS=(
    "efficientsam3_tinyvit.pt"
    "efficientsam3_efficientvit.pt"
    "efficientsam3_repvit.pt"
)

echo "=========================================================="
echo "Downloading EfficientSAM3 Pretrained Weights"
echo "Target directory: $DEST_DIR"
echo "=========================================================="

for model in "${MODELS[@]}"; do
    TARGET_PATH="$DEST_DIR/$model"
    if [ -f "$TARGET_PATH" ]; then
        echo "[SKIP] $model already exists at $TARGET_PATH"
    else
        echo "[DOWNLOADING] $model from $BASE_URL/$model..."
        if command -v curl &> /dev/null; then
            curl -L -f --progress-bar "$BASE_URL/$model" -o "$TARGET_PATH"
        elif command -v wget &> /dev/null; then
            wget -q --show-progress "$BASE_URL/$model" -O "$TARGET_PATH"
        else
            echo "ERROR: Neither curl nor wget found. Please install one to download."
            exit 1
        fi
        echo "[DONE] Downloaded $model"
    fi
done

echo "=========================================================="
echo "All checkpoints ready in $DEST_DIR"
echo "=========================================================="
