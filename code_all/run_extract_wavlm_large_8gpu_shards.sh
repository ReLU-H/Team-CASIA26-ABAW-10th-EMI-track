#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_SHARDS=8
OUT_DIR=/home/huangjiawen/EMI/test_set/test_audio_feats

for GID in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=${GID} python extract_wavlm_large_feats.py \
    --audio_dir /home/huangjiawen/EMI/test_set/audio_test_set/audio \
    --out_dir "${OUT_DIR}" \
    --model_name microsoft/wavlm-large \
    --device cuda \
    --batch_size 8 \
    --sample_rate 16000 \
    --max_seconds 0 \
    --resume 1 \
    --use_amp 1 \
    --use_safetensors 1 \
    --use_data_parallel 0 \
    --min_batch_size 1 \
    --oom_chunk_seconds 20 \
    --seed 42 \
    --shard_id ${GID} \
    --num_shards ${NUM_SHARDS} &
done

wait
echo "[OK] all shards finished"

