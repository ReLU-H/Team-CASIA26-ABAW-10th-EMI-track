#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/huangjiawen/.conda/envs/EMI/bin/python"
PYTHON_SCRIPT="/home/huangjiawen/EMI/extract_chatglm3_feats.py"
TRANSCRIPT_CSV="/home/huangjiawen/EMI/test_set/test_transcript/transcripts.csv"
OUT_DIR="/home/huangjiawen/EMI/test_set/test_text_feats"

# Example:
#   GPU_IDS="2 3 5 7" RESUME=1 STAGGER_SECONDS=2 bash run_extract_chatglm3_feats.sh
GPU_IDS_STR="${GPU_IDS:-0 1 2 3 4 5 6 7}"
RESUME="${RESUME:-1}"
STAGGER_SECONDS="${STAGGER_SECONDS:-2}"

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
NUM_SHARDS=${#GPU_IDS[@]}
PIDS=()

if [[ ${NUM_SHARDS} -eq 0 ]]; then
  echo "[ERROR] GPU_IDS is empty"
  exit 1
fi

echo "[INFO] transcript csv: ${TRANSCRIPT_CSV}"
echo "[INFO] output dir: ${OUT_DIR}"
echo "[INFO] gpu ids: ${GPU_IDS[*]}"
echo "[INFO] num shards: ${NUM_SHARDS}"
echo "[INFO] resume: ${RESUME}"
echo "[INFO] stagger seconds: ${STAGGER_SECONDS}"

for SHARD_ID in "${!GPU_IDS[@]}"; do
  GID="${GPU_IDS[$SHARD_ID]}"
  echo "[INFO] launch shard ${SHARD_ID}/${NUM_SHARDS} on physical GPU ${GID}"
  CUDA_VISIBLE_DEVICES="${GID}" "${PYTHON_BIN}" "${PYTHON_SCRIPT}" \
    --transcript_csv "${TRANSCRIPT_CSV}" \
    --out_dir "${OUT_DIR}" \
    --model_name THUDM/chatglm3-6b \
    --device cuda \
    --max_length 256 \
    --drop_special_tokens 1 \
    --resume "${RESUME}" \
    --use_amp 1 \
    --use_safetensors 1 \
    --seed 42 \
    --shard_id "${SHARD_ID}" \
    --num_shards "${NUM_SHARDS}" &
  PIDS+=($!)

  if (( SHARD_ID + 1 < NUM_SHARDS )) && [[ "${STAGGER_SECONDS}" != "0" ]]; then
    sleep "${STAGGER_SECONDS}"
  fi
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ ${FAIL} -ne 0 ]]; then
  echo "[ERROR] some shards failed"
  exit 1
fi

echo "[OK] all shards finished -> ${OUT_DIR}"

