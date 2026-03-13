#!/usr/bin/env bash
set -euo pipefail

# Phase 3 feature extraction in 8 shards (one process per GPU).
# It will generate vit/<id>.pkl, each file stores np.ndarray(float32) [T, D].

NUM_SHARDS=8
FACE_DIR="${FACE_DIR:-face_images}"
OUT_DIR="${OUT_DIR:-images_feats_pre-train}"
ENCODER_PATH="${ENCODER_PATH:-ckpts/finetune/EmoViT_encoder.pt}"
POOLING="${POOLING:-cls}"   # cls or mean
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "${OUT_DIR}"

for GID in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${GID}" python extract_emovit_feats.py \
    --face_dir "${FACE_DIR}" \
    --out_dir "${OUT_DIR}" \
    --encoder_path "${ENCODER_PATH}" \
    --image_size 224 \
    --id_width 5 \
    --pooling "${POOLING}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor 4 \
    --shard_id "${GID}" \
    --num_shards "${NUM_SHARDS}" \
    --device cuda \
    --bf16 1 &
done

wait
echo "[OK] all shards finished."

