#!/usr/bin/env bash
set -euo pipefail

# =========================
# ViT train + extract script
# =========================
#
# Usage:
#   bash run_vit_train_and_extract.sh
#
# Optional env vars:
#   TRAIN_ROOTS="/home/huangjiawen/.cache/huggingface/datasets/Piro17___affectnethq,/home/huangjiawen/.cache/huggingface/datasets/randomguyfromnepal___casia_web_face"
#   VAL_ROOTS="/path/to/val_affect,/path/to/val_casia"  # optional
#   FACE_DIR="/path/to/face_images"
#   OUT_DIR="outputs/vit_cls"
#   FEAT_DIR="images_feats_EmoVit"
#   CUDA_VISIBLE_DEVICES="0"

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TRAIN_ROOTS="${TRAIN_ROOTS:-/home/huangjiawen/.cache/huggingface/datasets/Piro17___affectnethq,/home/huangjiawen/.cache/huggingface/datasets/randomguyfromnepal___casia_web_face}"
VAL_ROOTS="${VAL_ROOTS:-}"
FACE_DIR="${FACE_DIR:-face_images}"
OUT_DIR="${OUT_DIR:-outputs/vit_cls}"
FEAT_DIR="${FEAT_DIR:-images_feats_EmoVit}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-256}"
BATCH_SIZE_EXTRACT="${BATCH_SIZE_EXTRACT:-256}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
BF16="${BF16:-1}"
MODEL_NAME="${MODEL_NAME:-vit_base_patch16_224}"
TRAIN_FORMAT="${TRAIN_FORMAT:-auto}"      # auto | folder | hf_cache
HF_SPLIT="${HF_SPLIT:-train}"
HF_IMAGE_COLUMN="${HF_IMAGE_COLUMN:-}"
HF_LABEL_COLUMN="${HF_LABEL_COLUMN:-}"

# echo "[1/3] Install dependencies..."
# "${PYTHON_BIN}" -m pip install -i "${PIP_INDEX_URL}" -U pip
# # IMPORTANT:
# # Keep numpy<2 to avoid pyarrow/pandas binary compatibility errors on this env.
# "${PYTHON_BIN}" -m pip install -i "${PIP_INDEX_URL}" -U --force-reinstall \
#   "numpy<2" \
#   "pyarrow>=14,<17" \
#   "pandas>=2.0,<2.3" \
#   "datasets>=2.14,<4.0" \
#   torch torchvision timm pillow tqdm

echo "[2/3] Train ViT classifier..."
if [[ -n "${VAL_ROOTS}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" vit_train_and_extract.py train \
    --train_roots "${TRAIN_ROOTS}" \
    --val_roots "${VAL_ROOTS}" \
    --train_format "${TRAIN_FORMAT}" \
    --hf_split "${HF_SPLIT}" \
    --hf_image_column "${HF_IMAGE_COLUMN}" \
    --hf_label_column "${HF_LABEL_COLUMN}" \
    --model_name "${MODEL_NAME}" \
    --image_size "${IMAGE_SIZE}" \
    --batch_size "${BATCH_SIZE_TRAIN}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --device "${DEVICE}" \
    --bf16 "${BF16}" \
    --output_dir "${OUT_DIR}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" vit_train_and_extract.py train \
    --train_roots "${TRAIN_ROOTS}" \
    --val_ratio 0.1 \
    --train_format "${TRAIN_FORMAT}" \
    --hf_split "${HF_SPLIT}" \
    --hf_image_column "${HF_IMAGE_COLUMN}" \
    --hf_label_column "${HF_LABEL_COLUMN}" \
    --model_name "${MODEL_NAME}" \
    --image_size "${IMAGE_SIZE}" \
    --batch_size "${BATCH_SIZE_TRAIN}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --device "${DEVICE}" \
    --bf16 "${BF16}" \
    --output_dir "${OUT_DIR}"
fi

echo "[3/3] Extract [CLS] features and save pkl..."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" vit_train_and_extract.py extract \
  --checkpoint "${OUT_DIR}/best_vit_cls.pth" \
  --face_dir "${FACE_DIR}" \
  --out_dir "${FEAT_DIR}" \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE_EXTRACT}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --bf16 "${BF16}" \
  --id_width 5

echo "[DONE] Train + extract completed."
