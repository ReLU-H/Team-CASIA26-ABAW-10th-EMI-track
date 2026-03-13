#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract WavLM-large audio features and save one pkl per sample ID.

Default output:
  test_audio_feats/<5-digit-id>.pkl

Each pkl stores a numpy.float32 array with shape [T, D], where:
  T = frame steps from WavLM hidden states
  D = model hidden size
"""

import argparse
import os
import pickle
import random
import subprocess
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".mp4", ".webm")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_id(x: Any, width: int = 5) -> str:
    s = str(x).strip()
    if s.isdigit():
        return s.zfill(width)
    try:
        return str(int(float(s))).zfill(width)
    except Exception:
        return s


def list_audio_files(audio_dir: str, exts: Sequence[str]) -> List[str]:
    files: List[str] = []
    for root, _, filenames in os.walk(audio_dir):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                files.append(os.path.join(root, name))
    files.sort()
    return files


def build_audio_map(audio_files: Sequence[str]) -> Dict[str, str]:
    audio_map: Dict[str, str] = {}
    for path in audio_files:
        sid = norm_id(os.path.splitext(os.path.basename(path))[0], 5)
        audio_map[sid] = path
    return audio_map


def shard_list(items: Sequence[str], shard_id: int, num_shards: int) -> List[str]:
    if num_shards <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]


def save_pkl(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_audio_mono(path: str, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
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


def maybe_trim_audio(audio: np.ndarray, sample_rate: int, max_seconds: float) -> np.ndarray:
    if max_seconds <= 0:
        return audio
    max_samples = int(sample_rate * max_seconds)
    if max_samples <= 0 or audio.shape[0] <= max_samples:
        return audio
    return audio[:max_samples]


def split_audio(audio: np.ndarray, chunk_seconds: float, sample_rate: int) -> List[np.ndarray]:
    if chunk_seconds <= 0:
        return [audio]
    chunk_samples = int(chunk_seconds * sample_rate)
    if chunk_samples <= 0 or audio.shape[0] <= chunk_samples:
        return [audio]
    chunks: List[np.ndarray] = []
    for start in range(0, audio.shape[0], chunk_samples):
        end = min(start + chunk_samples, audio.shape[0])
        part = audio[start:end]
        if part.size > 0:
            chunks.append(part)
    return chunks or [audio]


def collate_batch(
    batch_audio: Sequence[np.ndarray],
    processor: AutoFeatureExtractor,
    sample_rate: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    inputs = processor(
        list(batch_audio),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    out: Dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
    return out


def get_last_hidden(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        hidden = outputs.last_hidden_state
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        hidden = outputs[0]
    else:
        raise RuntimeError("Cannot parse model hidden states.")
    if hidden.dim() != 3:
        raise RuntimeError(f"Unexpected hidden shape: {tuple(hidden.shape)}")
    return hidden


def ensure_2d(arr: np.ndarray, hidden_size: int) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((1, hidden_size), dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def extract_batch_hidden(
    model: AutoModel,
    processor: AutoFeatureExtractor,
    batch_audio: Sequence[np.ndarray],
    sample_rate: int,
    device: str,
    use_amp: bool,
) -> List[np.ndarray]:
    enc = collate_batch(batch_audio, processor, sample_rate, device)
    amp_enabled = use_amp and device.startswith("cuda")
    amp_device = "cuda" if device.startswith("cuda") else "cpu"
    with torch.no_grad():
        with torch.amp.autocast(device_type=amp_device, enabled=amp_enabled):
            outputs = model(**enc, output_hidden_states=True, return_dict=True)
            hidden = get_last_hidden(outputs)

    attn = enc.get("attention_mask", None)
    feats: List[np.ndarray] = []
    if attn is None:
        for i in range(hidden.shape[0]):
            feats.append(hidden[i].float().cpu().numpy().astype(np.float32, copy=False))
        return feats

    valid_lens = attn.sum(dim=1).detach().cpu().numpy().astype(np.int64)
    raw_len = enc["input_values"].shape[1]
    hid_len = hidden.shape[1]
    ratio = float(hid_len) / float(max(raw_len, 1))

    for i in range(hidden.shape[0]):
        frame_len = int(np.ceil(float(valid_lens[i]) * ratio))
        frame_len = max(1, min(frame_len, hid_len))
        arr = hidden[i, :frame_len].float().cpu().numpy().astype(np.float32, copy=False)
        feats.append(arr)
    return feats


def extract_one_with_fallback(
    model: AutoModel,
    processor: AutoFeatureExtractor,
    audio: np.ndarray,
    sample_rate: int,
    device: str,
    use_amp: bool,
    hidden_size: int,
    oom_chunk_seconds: float,
) -> np.ndarray:
    try:
        return ensure_2d(
            extract_batch_hidden(
                model=model,
                processor=processor,
                batch_audio=[audio],
                sample_rate=sample_rate,
                device=device,
                use_amp=use_amp,
            )[0],
            hidden_size=hidden_size,
        )
    except torch.OutOfMemoryError:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        chunks = split_audio(audio, chunk_seconds=oom_chunk_seconds, sample_rate=sample_rate)
        if len(chunks) == 1:
            raise
        parts: List[np.ndarray] = []
        for chunk in chunks:
            part = extract_batch_hidden(
                model=model,
                processor=processor,
                batch_audio=[chunk],
                sample_rate=sample_rate,
                device=device,
                use_amp=use_amp,
            )[0]
            parts.append(ensure_2d(part, hidden_size=hidden_size))
        return ensure_2d(np.concatenate(parts, axis=0), hidden_size=hidden_size)


def flush_batch(
    batch: List[Tuple[str, np.ndarray]],
    model: AutoModel,
    processor: AutoFeatureExtractor,
    out_dir: str,
    sample_rate: int,
    device: str,
    use_amp: bool,
    hidden_size: int,
    min_batch_size: int,
    oom_chunk_seconds: float,
) -> Tuple[int, int]:
    if not batch:
        return 0, 0

    sids = [x[0] for x in batch]
    audios = [x[1] for x in batch]
    ok = 0
    fail = 0

    try:
        feats = extract_batch_hidden(
            model=model,
            processor=processor,
            batch_audio=audios,
            sample_rate=sample_rate,
            device=device,
            use_amp=use_amp,
        )
        for sid, arr in zip(sids, feats):
            save_pkl(os.path.join(out_dir, f"{sid}.pkl"), ensure_2d(arr, hidden_size=hidden_size))
            ok += 1
        return ok, fail
    except torch.OutOfMemoryError:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        if len(batch) > max(1, min_batch_size):
            mid = max(1, len(batch) // 2)
            ok1, fail1 = flush_batch(
                batch[:mid],
                model,
                processor,
                out_dir,
                sample_rate,
                device,
                use_amp,
                hidden_size,
                min_batch_size,
                oom_chunk_seconds,
            )
            ok2, fail2 = flush_batch(
                batch[mid:],
                model,
                processor,
                out_dir,
                sample_rate,
                device,
                use_amp,
                hidden_size,
                min_batch_size,
                oom_chunk_seconds,
            )
            return ok1 + ok2, fail1 + fail2

        for sid, audio in batch:
            out_path = os.path.join(out_dir, f"{sid}.pkl")
            try:
                arr = extract_one_with_fallback(
                    model=model,
                    processor=processor,
                    audio=audio,
                    sample_rate=sample_rate,
                    device=device,
                    use_amp=use_amp,
                    hidden_size=hidden_size,
                    oom_chunk_seconds=oom_chunk_seconds,
                )
                save_pkl(out_path, arr)
                ok += 1
            except Exception as e:
                save_pkl(out_path, np.zeros((1, hidden_size), dtype=np.float32))
                fail += 1
                print(f"[WARN] {sid} failed: {e}")
        return ok, fail
    except Exception:
        for sid, audio in batch:
            out_path = os.path.join(out_dir, f"{sid}.pkl")
            try:
                arr = extract_one_with_fallback(
                    model=model,
                    processor=processor,
                    audio=audio,
                    sample_rate=sample_rate,
                    device=device,
                    use_amp=use_amp,
                    hidden_size=hidden_size,
                    oom_chunk_seconds=oom_chunk_seconds,
                )
                save_pkl(out_path, arr)
                ok += 1
            except Exception as e:
                save_pkl(out_path, np.zeros((1, hidden_size), dtype=np.float32))
                fail += 1
                print(f"[WARN] {sid} failed: {e}")
        return ok, fail


def build_model(model_name: str, device: str, use_safetensors: bool, use_data_parallel: bool) -> Tuple[Any, Any]:
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model_kwargs = {"use_safetensors": use_safetensors}
    try:
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        model = AutoModel.from_pretrained(model_name)

    model = model.to(device).eval()
    if use_data_parallel and device.startswith("cuda") and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return processor, model


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract WavLM-large features as per-id pkl files.")
    ap.add_argument("--audio_dir", type=str, default="audio")
    ap.add_argument("--out_dir", type=str, default="test_audio_feats")
    ap.add_argument("--model_name", type=str, default="microsoft/wavlm-large")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--max_seconds", type=float, default=0)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--use_safetensors", type=int, default=1)
    ap.add_argument("--use_data_parallel", type=int, default=0)
    ap.add_argument("--min_batch_size", type=int, default=1)
    ap.add_argument("--oom_chunk_seconds", type=float, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--strict_count_check", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    audio_files = list_audio_files(args.audio_dir, AUDIO_EXTS)
    if not audio_files:
        raise RuntimeError(f"No audio files found in: {args.audio_dir}")
    audio_map = build_audio_map(audio_files)
    all_ids = sorted(audio_map.keys())
    shard_ids = shard_list(all_ids, args.shard_id, args.num_shards)

    print(f"[INFO] audio dir: {args.audio_dir}")
    print(f"[INFO] total audio files: {len(audio_files)}")
    print(f"[INFO] shard rows: {len(shard_ids)} (shard {args.shard_id}/{args.num_shards})")
    if len(shard_ids) == 0:
        print("[OK] nothing to process")
        return

    if bool(args.resume):
        before = len(shard_ids)
        shard_ids = [sid for sid in shard_ids if not os.path.exists(os.path.join(args.out_dir, f"{sid}.pkl"))]
        print(f"[INFO] resume filter: {before} -> {len(shard_ids)}")
        if len(shard_ids) == 0:
            print("[OK] all done for this shard")
            return

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = args.device
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    use_amp = bool(args.use_amp) and device.startswith("cuda")

    processor, model = build_model(
        model_name=args.model_name,
        device=device,
        use_safetensors=bool(args.use_safetensors),
        use_data_parallel=bool(args.use_data_parallel),
    )
    model_cfg = model.module.config if hasattr(model, "module") else model.config
    hidden_size = int(getattr(model_cfg, "hidden_size", 1024))

    ok = 0
    fail = 0
    pending: List[Tuple[str, np.ndarray]] = []
    pbar = tqdm(shard_ids, ncols=110, desc="Extract WavLM feats")

    for sid in pbar:
        path = audio_map[sid]
        out_path = os.path.join(args.out_dir, f"{sid}.pkl")
        try:
            audio = load_audio_mono(path, sample_rate=args.sample_rate)
            audio = maybe_trim_audio(audio, sample_rate=args.sample_rate, max_seconds=args.max_seconds)
            pending.append((sid, audio))
        except Exception as e:
            save_pkl(out_path, np.zeros((1, hidden_size), dtype=np.float32))
            fail += 1
            pbar.set_postfix(ok=ok, fail=fail)
            print(f"[WARN] {sid} failed: {e}")
            continue

        if len(pending) >= max(1, args.batch_size):
            b_ok, b_fail = flush_batch(
                batch=pending,
                model=model,
                processor=processor,
                out_dir=args.out_dir,
                sample_rate=args.sample_rate,
                device=device,
                use_amp=use_amp,
                hidden_size=hidden_size,
                min_batch_size=max(1, args.min_batch_size),
                oom_chunk_seconds=float(args.oom_chunk_seconds),
            )
            ok += b_ok
            fail += b_fail
            pending = []
            pbar.set_postfix(ok=ok, fail=fail)

    if pending:
        b_ok, b_fail = flush_batch(
            batch=pending,
            model=model,
            processor=processor,
            out_dir=args.out_dir,
            sample_rate=args.sample_rate,
            device=device,
            use_amp=use_amp,
            hidden_size=hidden_size,
            min_batch_size=max(1, args.min_batch_size),
            oom_chunk_seconds=float(args.oom_chunk_seconds),
        )
        ok += b_ok
        fail += b_fail
        pbar.set_postfix(ok=ok, fail=fail)

    print(f"[OK] output dir: {args.out_dir}")
    print(f"[OK] shard write done: ok={ok}, fail={fail}, total={ok + fail}")

    if bool(args.strict_count_check):
        expected = set(shard_ids)
        got = {sid for sid in expected if os.path.exists(os.path.join(args.out_dir, f"{sid}.pkl"))}
        miss = sorted(expected - got)
        print(f"[CHECK] shard expected={len(expected)}, got={len(got)}, missing={len(miss)}")
        if miss:
            raise RuntimeError(f"Missing feature files in shard, examples: {miss[:10]}")


if __name__ == "__main__":
    main()
