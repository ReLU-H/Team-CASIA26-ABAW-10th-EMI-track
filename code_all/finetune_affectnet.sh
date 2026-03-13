#!/usr/bin/env bash
set -euo pipefail

# 8x GPU DDP launch for Phase 2 AffectNetHQ finetuning.
# You can override env vars before running:
#   MAE_CKPT, OUTPUT_DIR, EXPORT_PATH, BATCH_SIZE, EPOCHS, LR

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export DDP_TIMEOUT_SECONDS="${DDP_TIMEOUT_SECONDS:-14400}"

# Auto infer number of processes from visible devices unless manually provided.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_GPU_ARR[@]}"
fi

MAE_CKPT="${MAE_CKPT:-ckpts/mae/checkpoint_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpts/finetune}"
EXPORT_PATH="${EXPORT_PATH:-ckpts/finetune/EmoViT_encoder.pt}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-5e-5}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port=29512 \
  finetune_emovit_affectnet.py \
  --dataset_name Piro17/affectnethq \
  --train_split train \
  --image_column "" \
  --label_column "" \
  --mae_ckpt "${MAE_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --export_encoder_path "${EXPORT_PATH}" \
  --image_size 224 \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num_workers 8 \
  --lr "${LR}" \
  --weight_decay 0.05 \
  --min_lr 1e-6 \
  --dropout 0.1 \
  --freeze_encoder_epochs 0 \
  --select_metric f1 \
  --dataset_download_retries 8 \
  --dataset_retry_wait_seconds 8 \
  --bf16 1

