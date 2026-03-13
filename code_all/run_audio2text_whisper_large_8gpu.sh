#!/usr/bin/env bash
set -euo pipefail

# --- User Config ---
PYTHON_BIN="/home/huangjiawen/.conda/envs/EMI/bin/python"
AUDIO_DIR="/home/huangjiawen/EMI/test_set/audio_test_set/audio"
OUT_DIR="/home/huangjiawen/EMI/test_set/test_transcript"
OUT_TXT_DIR="${OUT_DIR}/transcripts_txt"
# Recommended for standard SOTA
MODEL_NAME="openai/whisper-large-v3"
EXPECTED_CSVS="EMI_Train.csv,EMI_Val.csv,EMI_Test.csv"
STRICT_NON_EMPTY=1
LOCAL_FILES_ONLY=0
STARTUP_STAGGER_SEC=3

mkdir -p "${OUT_DIR}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[INFO] CUDA_VISIBLE_DEVICES not set, defaulting to 0,1,2,3,4,5,6,7"
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi

IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NUM_SHARDS=${#DEVICES[@]}

echo "======================================================"
echo " Starting Whisper ASR Extraction (Sharded)"
echo " Output Directory  : ${OUT_DIR}"
echo " GPU Devices       : ${CUDA_VISIBLE_DEVICES}"
echo " Total Shards      : ${NUM_SHARDS}"
echo " Model             : ${MODEL_NAME}"
echo "======================================================"

# Stabilize multi-process runtime (avoid oversubscription / native crashes)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Pre-download/check model artifacts once to reduce startup races.
if [[ "${LOCAL_FILES_ONLY}" -eq 0 ]]; then
  echo "[INFO] prefetch model cache: ${MODEL_NAME}"
  "${PYTHON_BIN}" - <<PY
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
name = "${MODEL_NAME}"
AutoProcessor.from_pretrained(name)
AutoModelForSpeechSeq2Seq.from_pretrained(name)
print("[OK] model cache ready")
PY
fi

PIDS=()
for idx in "${!DEVICES[@]}"; do
  dev="${DEVICES[$idx]}"
  shard_csv="${OUT_DIR}/transcripts_shard${idx}.csv"
  echo "[LAUNCH] shard=${idx}/${NUM_SHARDS} cuda=${dev}"
  CUDA_VISIBLE_DEVICES="${dev}" "${PYTHON_BIN}" audio2text_whisper_large.py \
    --audio_dir "${AUDIO_DIR}" \
    --out_csv "${shard_csv}" \
    --out_txt_dir "${OUT_TXT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --device cuda \
    --shard_id "${idx}" \
    --num_shards "${NUM_SHARDS}" \
    --resume 1 \
    --expected_ids_csvs "${EXPECTED_CSVS}" \
    --strict_expected_coverage 0 \
    --strict_non_empty 0 \
    --local_files_only "${LOCAL_FILES_ONLY}" &
  PIDS+=($!)
  sleep "${STARTUP_STAGGER_SEC}"
done

echo "[INFO] Wait for all ${NUM_SHARDS} shards to complete..."
FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ ${FAIL} -ne 0 ]]; then
  echo "[ERROR] One or more ASR shards failed."
  exit 1
fi

echo "[OK] All shards completed successfully. Merging results..."

# Python block to merge CSVs and extract empty IDs
"${PYTHON_BIN}" - <<PY
import os
import glob
import pandas as pd

out_dir = "${OUT_DIR}"
merged_csv = os.path.join(out_dir, "transcripts.csv")
empty_txt = os.path.join(out_dir, "empty_transcript_ids.txt")
empty_csv = os.path.join(out_dir, "empty_transcript_ids.csv")

shard_files = glob.glob(os.path.join(out_dir, "transcripts_shard*.csv"))
if not shard_files:
    print("[ERROR] No shard csv files found.")
    exit(1)

dfs = []
for f in sorted(shard_files):
    try:
        dfs.append(pd.read_csv(f))
    except Exception as e:
        print(f"[WARN] Failed to read {f}: {e}")

if not dfs:
    print("[ERROR] No valid CSV data to merge.")
    exit(1)

df = pd.concat(dfs, axis=0, ignore_index=True)
if "Filename" in df.columns:
    df["Filename"] = df["Filename"].astype(str).str.zfill(5)
    df = df.drop_duplicates(subset=["Filename"], keep="last")
    df = df.sort_values("Filename")

df.to_csv(merged_csv, index=False)
print(f"[OK] Merged {len(df)} rows into {merged_csv}")

# Identify empty or repaired transcripts
bad_mask = (df["text"].fillna("").str.strip() == "") | \
           (df["text"].fillna("").str.strip() == "[NO_SPEECH]") | \
           (df["repair_note"].fillna("").str.strip() != "") | \
           (df["error"].fillna("").str.strip() != "")

df_bad = df[bad_mask].copy()
if len(df_bad) > 0:
    print(f"[WARN] Found {len(df_bad)} records with empty/repaired transcripts. Saving to {empty_txt} & {empty_csv}")
    bad_ids = sorted(df_bad["Filename"].tolist())
    with open(empty_txt, "w") as f:
        for bid in bad_ids:
            f.write(str(bid) + "\n")
    df_bad.to_csv(empty_csv, index=False)
else:
    print("[OK] No empty or repaired transcripts found.")

if ${STRICT_NON_EMPTY} == 1 and len(df_bad) > 0:
    print(f"[ERROR] strict_non_empty check failed! {len(df_bad)} empty/repaired/error transcripts found.")
    exit(1)
PY

if [ $? -ne 0 ]; then
  echo "[ERROR] Merge step failed."
  exit 1
fi

echo "[OK] Whisper ASR processing complete!"
