#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR with Whisper Large model, sharded by ID.

Outputs:
  1) shard csv: transcript_text_whisper_large/transcripts_shard{n}.csv
  2) per-id txt: transcript_text_whisper_large/transcripts_txt/{id}.txt

Design goals:
  - shard by sample ID for stable multi-GPU split
  - strict coverage check against train/valid IDs
  - strict non-empty text check (fail fast if any empty)
"""

import argparse
import os
import subprocess
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def norm_id(x: Any, width: int = 5) -> str:
    s = str(x).strip()
    if s.isdigit():
        return s.zfill(width)
    try:
        return str(int(float(s))).zfill(width)
    except Exception:
        return s


def list_audio_files(audio_dir: str, exts: Tuple[str, ...]) -> List[str]:
    files = []
    for root, _, fnames in os.walk(audio_dir):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    files.sort()
    return files


def build_audio_map(audio_files: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in audio_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        out[norm_id(stem, 5)] = p
    return out


def load_expected_ids(csv_paths: str) -> List[str]:
    ids: Set[str] = set()
    for p in [x.strip() for x in str(csv_paths).split(",") if x.strip()]:
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if "Filename" not in df.columns:
            continue
        for x in df["Filename"].tolist():
            ids.add(norm_id(x, 5))
    return sorted(ids)


def shard_list(items: List[str], shard_id: int, num_shards: int) -> List[str]:
    if num_shards <= 1:
        return items
    return [x for i, x in enumerate(items) if (i % num_shards) == shard_id]


def safe_write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write((text or "").strip() + "\n")


def load_audio_16k_mono(path: str) -> np.ndarray:
    """
    Decode audio with ffmpeg to avoid torchaudio/librosa backend segfaults
    under heavy multi-process GPU runs.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-i", path,
        "-ac", "1",
        "-ar", "16000",
        "-f", "s16le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        err = ""
        if hasattr(e, "stderr") and e.stderr:
            try:
                err = e.stderr.decode("utf-8", errors="ignore").strip()
            except Exception:
                err = str(e)
        raise RuntimeError(f"ffmpeg_decode_failed: {err or e}")

    if not proc.stdout:
        raise RuntimeError("ffmpeg_decode_failed: empty_pcm")
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise RuntimeError("ffmpeg_decode_failed: empty_audio")
    return audio


@torch.no_grad()
def transcribe_one(
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    audio_np: np.ndarray,
    device: str,
) -> str:
    # Whisper processor
    inputs = processor(
        audio_np, 
        sampling_rate=16000, 
        return_tensors="pt"
    )
    # Whisper inference requires matching dtype (e.g., float16)
    input_features = inputs.input_features.to(device, dtype=model.dtype)
    
    generated_ids = model.generate(
        input_features,
        max_new_tokens=256
    )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return (text or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default="audio")
    ap.add_argument("--out_csv", default="transcript_text_whisper_large/transcripts.csv")
    ap.add_argument("--out_txt_dir", default="transcript_text_whisper_large/transcripts_txt")
    ap.add_argument("--model_name", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--expected_ids_csvs", default="train_split.csv,valid_split.csv")
    ap.add_argument("--strict_expected_coverage", type=int, default=1)
    ap.add_argument("--strict_non_empty", type=int, default=1)
    ap.add_argument("--local_files_only", type=int, default=0)
    ap.add_argument("--repair_empty_text", type=int, default=1)
    ap.add_argument("--empty_text_fallback", type=str, default="[NO_SPEECH]")
    ap.add_argument("--silence_rms_threshold", type=float, default=1e-4)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(args.out_txt_dir, exist_ok=True)

    exts = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".mp4", ".webm")
    audio_files = list_audio_files(args.audio_dir, exts)
    if not audio_files:
        raise RuntimeError(f"No audio files found in {args.audio_dir}")
    audio_map = build_audio_map(audio_files)

    expected_ids = load_expected_ids(args.expected_ids_csvs)
    if expected_ids:
        items = expected_ids
    else:
        items = sorted(audio_map.keys())
    items = shard_list(items, args.shard_id, args.num_shards)

    done_ids: Set[str] = set()
    if bool(args.resume) and os.path.exists(args.out_csv):
        try:
            old = pd.read_csv(args.out_csv)
            if "Filename" in old.columns:
                done_ids = set(old["Filename"].astype(str).apply(lambda x: norm_id(x, 5)).tolist())
        except Exception:
            done_ids = set()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    print(f"[INFO] device={device}, shard={args.shard_id}/{args.num_shards}, items={len(items)}")
    print(f"[INFO] model={args.model_name}")

    local_only = bool(args.local_files_only)
    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=local_only)
    
    # Use torch.float16 for Whisper on CUDA to save VRAM and drastically speed up inference
    target_dtype = torch.float16 if device == "cuda" else torch.float32
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name, 
        torch_dtype=target_dtype,
        local_files_only=local_only
    ).to(device).eval()

    rows: List[Dict[str, Any]] = []
    pbar = tqdm(items, desc=f"Whisper-ASR {args.shard_id}/{args.num_shards}", ncols=100)
    for sid in pbar:
        sid = norm_id(sid, 5)
        if bool(args.resume) and sid in done_ids:
            continue
        path = audio_map.get(sid, "")
        if not path:
            rows.append({
                "Filename": sid,
                "text": "",
                "audio_path": "",
                "model": args.model_name,
                "error": "audio_not_found",
            })
            continue
        try:
            audio_np = load_audio_16k_mono(path)
            rms = float(np.sqrt(np.mean(np.square(audio_np))) + 1e-12)
            is_silent = rms < float(args.silence_rms_threshold)
            text = transcribe_one(model, processor, audio_np, device)
            repair_reason = ""
            if not text:
                if bool(args.repair_empty_text):
                    if is_silent:
                        repair_reason = "silent_audio_repaired"
                    else:
                        repair_reason = "empty_transcript_repaired"
                    text = str(args.empty_text_fallback).strip() or "[NO_SPEECH]"
                else:
                    raise RuntimeError("empty_transcript")
            safe_write_text(os.path.join(args.out_txt_dir, f"{sid}.txt"), text)
            rows.append({
                "Filename": sid,
                "text": text,
                "audio_path": path,
                "model": args.model_name,
                "error": "",
                "repair_note": repair_reason,
                "audio_rms": rms,
            })
        except Exception as e:
            rows.append({
                "Filename": sid,
                "text": "",
                "audio_path": path,
                "model": args.model_name,
                "error": str(e),
                "repair_note": "",
                "audio_rms": float("nan"),
            })

    df_new = pd.DataFrame(rows)
    if os.path.exists(args.out_csv):
        try:
            df_old = pd.read_csv(args.out_csv)
            df_all = pd.concat([df_old, df_new], axis=0, ignore_index=True)
            if "Filename" in df_all.columns:
                df_all["Filename"] = df_all["Filename"].astype(str).apply(lambda x: norm_id(x, 5))
                df_all = df_all.drop_duplicates(subset=["Filename"], keep="last")
        except Exception:
            df_all = df_new
    else:
        df_all = df_new

    for c in ["Filename", "text", "audio_path", "model", "error", "repair_note", "audio_rms"]:
        if c not in df_all.columns:
            df_all[c] = ""
    df_all = df_all[["Filename", "text", "audio_path", "model", "error", "repair_note", "audio_rms"]]
    df_all.to_csv(args.out_csv, index=False)
    print(f"[OK] saved: {args.out_csv} rows={len(df_all)}")

    if expected_ids and bool(args.strict_expected_coverage):
        got_set = set(df_all["Filename"].astype(str).apply(lambda x: norm_id(x, 5)).tolist())
        miss = sorted(set(expected_ids) - got_set)
        if miss:
            raise RuntimeError(f"coverage_failed: missing {len(miss)} ids, examples={miss[:10]}")

    if bool(args.strict_non_empty):
        text_series = df_all["text"].fillna("").astype(str).str.strip()
        err_series = df_all["error"].fillna("").astype(str).str.strip()
        empty_cnt = int((text_series == "").sum())
        err_cnt = int((err_series != "").sum())
        repaired_cnt = int((df_all["repair_note"].fillna("").astype(str).str.strip() != "").sum()) if "repair_note" in df_all.columns else 0
        if repaired_cnt > 0:
            print(f"[INFO] repaired_empty_text={repaired_cnt} (fallback='{args.empty_text_fallback}')")
        if empty_cnt > 0 or err_cnt > 0:
            bad = df_all[(text_series == "") | (err_series != "")].head(10)
            print("[ERROR] empty/error examples:")
            print(bad[["Filename", "text", "error"]].to_string(index=False))
            raise RuntimeError(f"strict_non_empty_failed: empty={empty_cnt}, error={err_cnt}")
        print("[OK] strict_non_empty passed (all transcripts non-empty)")


if __name__ == "__main__":
    main()
