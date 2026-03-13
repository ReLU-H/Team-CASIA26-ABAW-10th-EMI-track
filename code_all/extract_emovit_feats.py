#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Batch visual feature extraction for EMI multimodal training.

核心要求:
1) 输出为 vit/<5-digit-id>.pkl
2) 一个 ID 多帧 => [T, D], 单帧 => [1, D]
3) pickle.dump(arr, protocol=pickle.HIGHEST_PROTOCOL), arr 必须是 numpy.float32
4) pooling 默认 cls, 支持 --pooling mean
5) 已存在文件自动跳过，支持断点续跑
"""

import argparse
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTMAEConfig, ViTMAEModel

from emovit_common import build_eval_transform, extract_id_from_path, normalize_id


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_all_images(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMAGE_EXTS:
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def natural_key(path: str):
    # 让帧排序更稳定，例如 00000_1.jpg, 00000_2.jpg, ... 00000_10.jpg
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]


def group_images_by_id(image_paths: List[str], id_width: int = 5) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for p in image_paths:
        raw_id: Optional[str] = extract_id_from_path(p)
        if raw_id is None:
            continue
        sid = normalize_id(raw_id, width=id_width)
        groups[sid].append(p)

    # Sort frames inside each ID group in natural order.
    for sid in groups:
        groups[sid].sort(key=natural_key)
    return groups


def load_encoder(encoder_path: str, device: torch.device) -> ViTMAEModel:
    obj = torch.load(encoder_path, map_location="cpu")
    if isinstance(obj, dict) and "encoder_state_dict" in obj:
        cfg_dict = obj.get("config", {})
        config = ViTMAEConfig(**cfg_dict) if cfg_dict else ViTMAEConfig()
        state = obj["encoder_state_dict"]
    else:
        # 兼容直接存 state_dict 的情况
        config = ViTMAEConfig()
        state = obj

    encoder = ViTMAEModel(config)
    msg = encoder.load_state_dict(state, strict=False)
    print(
        f"[LOAD] encoder from {encoder_path} | missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}",
        flush=True,
    )
    encoder.eval()
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def shard_ids(ids: List[str], shard_id: int, num_shards: int) -> List[str]:
    if num_shards <= 1:
        return ids
    return [sid for i, sid in enumerate(ids) if i % num_shards == shard_id]


@dataclass
class FrameEntry:
    sid: str
    path: str


class FrameDataset(Dataset):
    def __init__(self, entries: List[FrameEntry], transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        img = Image.open(e.path).convert("RGB")
        x = self.transform(img)
        return x, e.sid


def collate_frames(batch):
    xs, sids = zip(*batch)
    return torch.stack(xs, dim=0), list(sids)


def main():
    ap = argparse.ArgumentParser(description="Extract EmoViT visual features as pkl numpy arrays.")
    ap.add_argument("--face_dir", type=str, default="face_images")
    ap.add_argument("--out_dir", type=str, default="vit")
    ap.add_argument("--encoder_path", type=str, default="ckpts/finetune/EmoViT_encoder.pt")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--id_width", type=int, default=5)
    ap.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    transform = build_eval_transform(args.image_size)

    image_paths = list_all_images(args.face_dir)
    groups = group_images_by_id(image_paths, id_width=args.id_width)
    all_ids = sorted(groups.keys())
    this_ids = shard_ids(all_ids, args.shard_id, args.num_shards)

    print(
        f"[INFO] total_images={len(image_paths)} total_ids={len(all_ids)} "
        f"shard={args.shard_id}/{args.num_shards} ids_in_shard={len(this_ids)}",
        flush=True,
    )

    # Resume: skip IDs that already exist.
    todo_ids: List[str] = []
    skipped = 0
    for sid in this_ids:
        if os.path.exists(os.path.join(args.out_dir, f"{sid}.pkl")):
            skipped += 1
        else:
            todo_ids.append(sid)
    print(f"[INFO] todo_ids={len(todo_ids)} skipped_existing={skipped}", flush=True)
    if not todo_ids:
        print(f"[DONE] extracted=0, skipped_existing={skipped}, out_dir={args.out_dir}", flush=True)
        return

    # Build frame list for global batching to maximize GPU utilization.
    entries: List[FrameEntry] = []
    for sid in todo_ids:
        for p in groups[sid]:
            entries.append(FrameEntry(sid=sid, path=p))
    print(f"[INFO] shard_frames={len(entries)} batch_size={args.batch_size}", flush=True)

    ds = FrameDataset(entries=entries, transform=transform)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate_frames,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(ds, **loader_kwargs)

    encoder = load_encoder(args.encoder_path, device=device)
    sid_feats: Dict[str, List[np.ndarray]] = defaultdict(list)
    use_bf16 = bool(args.bf16) and device.type == "cuda"

    with torch.inference_mode():
        for x, sids in tqdm(loader, desc="Infer", ncols=100):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = encoder(pixel_values=x)
                tokens = out.last_hidden_state  # [B, N, D]
                feat = tokens.mean(dim=1) if args.pooling == "mean" else tokens[:, 0, :]
            arr = feat.float().cpu().numpy().astype(np.float32, copy=False)
            for i, sid in enumerate(sids):
                sid_feats[sid].append(arr[i : i + 1])

    done = 0
    for sid in tqdm(todo_ids, desc="Save", ncols=100):
        out_path = os.path.join(args.out_dir, f"{sid}.pkl")
        feat = np.concatenate(sid_feats[sid], axis=0).astype(np.float32, copy=False)
        with open(out_path, "wb") as f:
            # 必须直接保存 np.ndarray，不要包成 dict
            pickle.dump(feat, f, protocol=pickle.HIGHEST_PROTOCOL)
        done += 1

    print(f"[DONE] extracted={done}, skipped_existing={skipped}, out_dir={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

