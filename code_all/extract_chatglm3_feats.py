#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ChatGLM3 sequence features from transcript CSV.

Default input:
  transcript_text_whisper_Large/transcripts.csv

Default output:
  text_feats_ChatGLM3/<5-digit-id>.pkl
  each pkl is np.ndarray(float32), shape [T, D]
"""

import argparse
import os
import pickle
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


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


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def shard_list(items: List[Dict[str, str]], shard_id: int, num_shards: int) -> List[Dict[str, str]]:
    if num_shards <= 1:
        return items
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]


def resolve_transcript_csv(transcript_csv: str, transcript_dir: str) -> str:
    if transcript_csv.strip():
        return transcript_csv
    return os.path.join(transcript_dir, "transcripts.csv")


def load_rows(csv_path: str, transcript_txt_dir: str = "") -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        raise RuntimeError(f"Transcript CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Filename" not in df.columns:
        raise RuntimeError(f"{csv_path} must contain column: Filename")
    if "text" not in df.columns:
        raise RuntimeError(f"{csv_path} must contain column: text")

    df["Filename"] = df["Filename"].apply(lambda x: norm_id(x, 5))
    df["text"] = df["text"].apply(safe_text)
    df = df.drop_duplicates(subset=["Filename"], keep="last").reset_index(drop=True)

    rows = []
    for sid, txt in zip(df["Filename"].tolist(), df["text"].tolist()):
        text_val = txt
        # Fallback to per-file transcript txt when CSV text is empty.
        if not text_val and transcript_txt_dir:
            txt_path = os.path.join(transcript_txt_dir, f"{sid}.txt")
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text_val = f.read().strip()
                except Exception:
                    text_val = ""
        rows.append({"sid": sid, "text": text_val})
    return rows


def patch_chatglm_config(cfg: Any, fallback_len: int) -> Any:
    if not hasattr(cfg, "max_length"):
        if hasattr(cfg, "seq_length"):
            cfg.max_length = int(getattr(cfg, "seq_length"))
        elif hasattr(cfg, "max_position_embeddings"):
            cfg.max_length = int(getattr(cfg, "max_position_embeddings"))
        else:
            cfg.max_length = int(fallback_len)
    if not hasattr(cfg, "use_cache"):
        cfg.use_cache = True
    return cfg


def get_hidden(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        h = outputs.last_hidden_state
    elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
        h = outputs.hidden_states[-1]
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        h = outputs[0]
    else:
        raise RuntimeError("Cannot parse hidden states from model output.")

    if h.dim() != 3:
        raise RuntimeError(f"Unexpected hidden tensor shape: {tuple(h.shape)}")

    # Some ChatGLM implementations return [T, B, D], convert to [B, T, D].
    if h.shape[1] == 1 and h.shape[0] > 1:
        h = h.transpose(0, 1).contiguous()
    return h


def get_special_keep_mask(tokenizer: Any, text: str, max_length: int) -> np.ndarray:
    sm = tokenizer(
        text,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_length,
    ).get("special_tokens_mask", None)
    if sm is None:
        return np.ones((0,), dtype=bool)
    return np.array(sm, dtype=np.int64) == 0


def save_pkl(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)


def ensure_2d(arr: np.ndarray, hidden_size: int) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((1, hidden_size), dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def patch_transformers_chatglm_compat() -> None:
    # Compatibility patch for newer transformers + older ChatGLM remote code.
    from transformers import modeling_utils

    cls = modeling_utils.PreTrainedModel
    if getattr(cls, "_emi_chatglm_tied_patch", False):
        return

    original_adjust = cls._adjust_tied_keys_with_tied_pointers
    original_mark = cls.mark_tied_weights_as_initialized

    def wrapped_adjust(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys") or self.all_tied_weights_keys is None:
            base_keys = getattr(self, "_tied_weights_keys", []) or []
            # Newer transformers expects a dict-like object with .keys()
            self.all_tied_weights_keys = {k: True for k in base_keys}
        elif isinstance(self.all_tied_weights_keys, (set, list, tuple)):
            self.all_tied_weights_keys = {k: True for k in self.all_tied_weights_keys}
        return original_adjust(self, *args, **kwargs)

    def wrapped_mark(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys") or self.all_tied_weights_keys is None:
            base_keys = getattr(self, "_tied_weights_keys", []) or []
            self.all_tied_weights_keys = {k: True for k in base_keys}
        elif isinstance(self.all_tied_weights_keys, (set, list, tuple)):
            self.all_tied_weights_keys = {k: True for k in self.all_tied_weights_keys}
        return original_mark(self, *args, **kwargs)

    cls._adjust_tied_keys_with_tied_pointers = wrapped_adjust
    cls.mark_tied_weights_as_initialized = wrapped_mark
    cls._emi_chatglm_tied_patch = True


def build_model(model_name: str, device: str, max_length: int, use_safetensors: bool) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    cfg = patch_chatglm_config(cfg, fallback_len=max_length)
    patch_transformers_chatglm_compat()

    target_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model_kwargs = dict(
        config=cfg,
        trust_remote_code=True,
        use_safetensors=use_safetensors,
    )
    try:
        model = AutoModel.from_pretrained(
            model_name,
            dtype=target_dtype,
            **model_kwargs,
        ).to(device)
    except TypeError:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=target_dtype,
            **model_kwargs,
        ).to(device)
    model.eval()
    return tokenizer, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript_csv", type=str, default="")
    ap.add_argument("--transcript_dir", type=str, default="transcript_text_whisper_Large")
    ap.add_argument("--transcript_txt_dir", type=str, default="transcript_text_whisper_Large/transcripts_txt")
    ap.add_argument("--out_dir", type=str, default="text_feats_ChatGLM3")
    ap.add_argument("--model_name", type=str, default="THUDM/chatglm3-6b")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--drop_special_tokens", type=int, default=1)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--use_safetensors", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--strict_count_check", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = resolve_transcript_csv(args.transcript_csv, args.transcript_dir)
    rows_all = load_rows(csv_path, transcript_txt_dir=args.transcript_txt_dir)
    rows = shard_list(rows_all, args.shard_id, args.num_shards)
    print(f"[INFO] transcript csv: {csv_path}")
    print(f"[INFO] shard rows: {len(rows)} (shard {args.shard_id}/{args.num_shards})")
    if len(rows) == 0:
        print("[OK] nothing to process")
        return

    if bool(args.resume):
        before = len(rows)
        rows = [r for r in rows if not os.path.exists(os.path.join(args.out_dir, f"{r['sid']}.pkl"))]
        print(f"[INFO] resume filter: {before} -> {len(rows)}")
        if len(rows) == 0:
            print("[OK] all done for this shard")
            return

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = args.device
    else:
        device = "cpu"
    use_amp = bool(args.use_amp) and device.startswith("cuda")
    amp_device = "cuda" if device.startswith("cuda") else "cpu"

    tokenizer, model = build_model(
        model_name=args.model_name,
        device=device,
        max_length=args.max_length,
        use_safetensors=bool(args.use_safetensors),
    )
    hidden_size = int(getattr(model.config, "hidden_size", 4096))

    ok = 0
    fail = 0
    pbar = tqdm(rows, ncols=110, desc="Extract ChatGLM3 feats")
    with torch.no_grad():
        for row in pbar:
            sid = row["sid"]
            text = row["text"] if row["text"] else "[EMPTY]"
            out_path = os.path.join(args.out_dir, f"{sid}.pkl")
            try:
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    out = model(**enc, output_hidden_states=True, return_dict=True)
                    hid = get_hidden(out)

                arr = hid[0].float().cpu().numpy().astype(np.float32)
                if bool(args.drop_special_tokens):
                    keep = get_special_keep_mask(tokenizer, text, args.max_length)
                    if keep.shape[0] == arr.shape[0] and keep.any():
                        arr = arr[keep]

                arr = ensure_2d(arr, hidden_size)
                save_pkl(out_path, arr)
                ok += 1
                pbar.set_postfix(ok=ok, fail=fail)
            except Exception as e:
                fallback = np.zeros((1, hidden_size), dtype=np.float32)
                save_pkl(out_path, fallback)
                fail += 1
                pbar.set_postfix(ok=ok, fail=fail)
                print(f"[WARN] {sid} failed: {e}")

    print(f"[OK] output dir: {args.out_dir}")
    print(f"[OK] shard write done: ok={ok}, fail={fail}, total={ok + fail}")

    if bool(args.strict_count_check):
        shard_sids = {r["sid"] for r in rows}
        got_sids = set()
        for sid in shard_sids:
            if os.path.exists(os.path.join(args.out_dir, f"{sid}.pkl")):
                got_sids.add(sid)
        miss = sorted(shard_sids - got_sids)
        print(f"[CHECK] shard expected={len(shard_sids)}, got={len(got_sids)}, missing={len(miss)}")
        if miss:
            raise RuntimeError(f"Missing feature files in shard, examples: {miss[:10]}")


if __name__ == "__main__":
    main()
