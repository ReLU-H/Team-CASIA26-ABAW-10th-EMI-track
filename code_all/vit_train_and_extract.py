#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViT training and feature extraction pipeline.

Features:
1) Generic PyTorch Dataset for folder-style classification datasets
   (compatible with AffectNetHQ / CASIA-WebFace in standard layout).
2) Fine-tune pretrained ViT-Base (timm) with automatic classifier output size.
3) Save best checkpoint during training.
4) Extract [CLS] features from face_images and save as per-id pkl:
   images_feats_EmoVit/<id>.pkl, each file stores np.ndarray(float32) [T, D].
"""

import argparse
import bisect
import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@dataclass
class Sample:
    path: str
    label: int


class FolderClassificationDataset(Dataset):
    """
    Generic dataset for one or more roots in:
      root/class_x/*.jpg
      root/class_y/*.png
    """

    def __init__(
        self,
        roots: Sequence[str],
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.roots = [str(Path(r).expanduser().resolve()) for r in roots]
        self.transform = transform
        self.class_to_idx = class_to_idx or self._build_global_class_map(self.roots)
        self.samples = self._collect_samples(self.roots, self.class_to_idx)
        if not self.samples:
            raise RuntimeError(f"No images found in dataset roots: {self.roots}")

    @staticmethod
    def _build_global_class_map(roots: Sequence[str]) -> Dict[str, int]:
        classes = set()
        for root in roots:
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if os.path.isdir(p):
                    classes.add(name)
        if not classes:
            raise RuntimeError(f"No class folders found in roots: {roots}")
        sorted_classes = sorted(classes)
        return {c: i for i, c in enumerate(sorted_classes)}

    @staticmethod
    def _collect_samples(roots: Sequence[str], class_to_idx: Dict[str, int]) -> List[Sample]:
        items: List[Sample] = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for cls_name, cls_idx in class_to_idx.items():
                cls_dir = os.path.join(root, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for dirpath, _, filenames in os.walk(cls_dir):
                    for fn in filenames:
                        if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                            items.append(Sample(path=os.path.join(dirpath, fn), label=cls_idx))
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img) if self.transform is not None else transforms.ToTensor()(img)
        return x, s.label


def maybe_to_pil(image_obj):
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, np.ndarray):
        arr = image_obj
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if isinstance(image_obj, dict):
        if "bytes" in image_obj and image_obj["bytes"] is not None:
            from io import BytesIO

            return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
        if "path" in image_obj and image_obj["path"]:
            return Image.open(image_obj["path"]).convert("RGB")
    if isinstance(image_obj, str):
        return Image.open(image_obj).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_obj)}")


def detect_image_column(ds_split, override: str = "") -> str:
    if override:
        return override
    names = list(getattr(ds_split, "column_names", []))
    lowered = {n.lower(): n for n in names}
    for k in ("image", "img", "pixel_values", "face", "jpg", "png"):
        if k in lowered:
            return lowered[k]
    for n in names:
        ln = n.lower()
        if "image" in ln or "img" in ln:
            return n
    raise RuntimeError(f"Cannot detect image column from: {names}")


def detect_label_column(ds_split, override: str = "") -> str:
    if override:
        return override
    names = list(getattr(ds_split, "column_names", []))
    lowered = {n.lower(): n for n in names}
    for k in ("label", "labels", "emotion", "expression", "identity", "id", "target", "category"):
        if k in lowered:
            return lowered[k]
    for n in names:
        ln = n.lower()
        if "label" in ln or "emotion" in ln or "expression" in ln or "identity" in ln:
            return n
    raise RuntimeError(f"Cannot detect label column from: {names}")


def find_hf_arrow_files(cache_root: str, split: str = "train") -> List[str]:
    root = str(Path(cache_root).expanduser().resolve())
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".arrow") and f"-{split}-" in fn:
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


class HFCachedArrowClassificationDataset(Dataset):
    """
    Classification dataset built from HuggingFace cached arrow files.
    Can combine multiple cache roots (e.g., AffectNetHQ + CASIA-WebFace).
    """

    def __init__(
        self,
        roots: Sequence[str],
        transform: transforms.Compose,
        split: str = "train",
        class_to_idx: Optional[Dict[str, int]] = None,
        image_column: str = "",
        label_column: str = "",
    ) -> None:
        self.transform = transform
        self.sources = []
        self.class_to_idx = {} if class_to_idx is None else class_to_idx
        self._lengths: List[int] = []
        try:
            from datasets import load_dataset  # local import: only needed for hf_cache mode
        except Exception as e:
            raise RuntimeError(
                "Failed to import 'datasets' stack. Please ensure compatible versions of "
                "numpy/pyarrow/pandas/datasets are installed."
            ) from e

        for root in roots:
            abs_root = str(Path(root).expanduser().resolve())
            data_files = find_hf_arrow_files(abs_root, split=split)
            if not data_files:
                raise RuntimeError(f"No arrow files for split='{split}' under: {abs_root}")
            ds = load_dataset("arrow", data_files={"train": data_files}, split="train")
            img_col = detect_image_column(ds, image_column)
            lbl_col = detect_label_column(ds, label_column)
            source_tag = os.path.basename(abs_root.rstrip("/")) or "dataset"

            # Build/extend unified class map with namespaced labels.
            if class_to_idx is None:
                uniq_labels = sorted(set(ds[lbl_col]))
                for lab in uniq_labels:
                    key = f"{source_tag}::{lab}"
                    if key not in self.class_to_idx:
                        self.class_to_idx[key] = len(self.class_to_idx)

            self.sources.append(
                {
                    "tag": source_tag,
                    "ds": ds,
                    "img_col": img_col,
                    "lbl_col": lbl_col,
                    "len": len(ds),
                }
            )
            self._lengths.append(len(ds))

        self._cum = np.cumsum(self._lengths).tolist()
        self.total_len = int(sum(self._lengths))
        if self.total_len <= 0:
            raise RuntimeError("Loaded HF cached datasets are empty.")

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        src_i = bisect.bisect_right(self._cum, idx)
        prev = 0 if src_i == 0 else self._cum[src_i - 1]
        local_idx = idx - prev
        src = self.sources[src_i]
        row = src["ds"][local_idx]
        image = maybe_to_pil(row[src["img_col"]])
        label_raw = row[src["lbl_col"]]
        label_key = f"{src['tag']}::{label_raw}"
        label = self.class_to_idx[label_key]
        x = self.transform(image)
        return x, label


class FaceImagesDataset(Dataset):
    """Dataset used for feature extraction from face_images."""

    def __init__(self, face_dir: str, transform: transforms.Compose) -> None:
        self.face_dir = str(Path(face_dir).expanduser().resolve())
        self.transform = transform
        self.image_paths: List[str] = []
        for dirpath, _, filenames in os.walk(self.face_dir):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                    self.image_paths.append(os.path.join(dirpath, fn))
        self.image_paths.sort()
        if not self.image_paths:
            raise RuntimeError(f"No images found in face_dir: {self.face_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, p


def normalize_id(raw: str, width: int = 5) -> str:
    s = str(raw).strip()
    if s.isdigit():
        return s.zfill(width)
    try:
        return str(int(float(s))).zfill(width)
    except Exception:
        m = re.search(r"\d+", s)
        if m:
            return m.group(0).zfill(width)
    return s


def extract_id_from_path(path: str) -> Optional[str]:
    # Prefer parent directory numeric id, then filename numeric id.
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    if len(parts) > 1:
        m0 = re.search(r"\d+", parts[-2])
        if m0:
            return m0.group(0)
    stem = os.path.splitext(parts[-1])[0]
    m1 = re.search(r"\d+", stem)
    if m1:
        return m1.group(0)
    return None


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def extract_cls_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Return CLS token features for ViT models.
    Compatible with common timm ViT outputs.
    """
    core_model = unwrap_model(model)
    feats = core_model.forward_features(x)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    if isinstance(feats, dict):
        if "x" in feats:
            feats = feats["x"]
        elif "last_hidden_state" in feats:
            feats = feats["last_hidden_state"]
        else:
            raise RuntimeError(f"Unsupported feature dict keys: {list(feats.keys())}")
    if feats.ndim == 3:
        return feats[:, 0, :]
    if feats.ndim == 2:
        return feats
    raise RuntimeError(f"Unsupported forward_features shape: {tuple(feats.shape)}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_bf16: bool) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    mean_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return mean_loss, acc


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    use_bf16 = bool(args.bf16) and device.type == "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    roots = [x.strip() for x in args.train_roots.split(",") if x.strip()]
    if not roots:
        raise ValueError("--train_roots is empty")

    def is_hf_cache_root(root: str) -> bool:
        return len(find_hf_arrow_files(root, split=args.hf_split)) > 0

    if args.train_format == "auto":
        has_hf = any(is_hf_cache_root(r) for r in roots)
        has_folder = any(os.path.isdir(r) and not is_hf_cache_root(r) for r in roots)
        if has_hf and has_folder:
            raise RuntimeError("Mixed train_roots types detected. Use all folder roots or all HF cache roots.")
        mode = "hf_cache" if has_hf else "folder"
    else:
        mode = args.train_format

    if mode == "hf_cache":
        base_ds = HFCachedArrowClassificationDataset(
            roots=roots,
            transform=build_eval_transform(args.image_size),
            split=args.hf_split,
            class_to_idx=None,
            image_column=args.hf_image_column,
            label_column=args.hf_label_column,
        )
    else:
        base_ds = FolderClassificationDataset(roots=roots, transform=build_eval_transform(args.image_size))

    num_classes = len(base_ds.class_to_idx)
    print(f"[DATA] mode={mode} train_roots={roots} classes={num_classes} images={len(base_ds)}", flush=True)

    if args.val_roots:
        val_roots = [x.strip() for x in args.val_roots.split(",") if x.strip()]
        if mode == "hf_cache":
            train_ds = HFCachedArrowClassificationDataset(
                roots=roots,
                transform=build_train_transform(args.image_size),
                split=args.hf_split,
                class_to_idx=base_ds.class_to_idx,
                image_column=args.hf_image_column,
                label_column=args.hf_label_column,
            )
            val_ds = HFCachedArrowClassificationDataset(
                roots=val_roots,
                transform=build_eval_transform(args.image_size),
                split=args.hf_split,
                class_to_idx=base_ds.class_to_idx,
                image_column=args.hf_image_column,
                label_column=args.hf_label_column,
            )
        else:
            train_ds = FolderClassificationDataset(
                roots=roots,
                transform=build_train_transform(args.image_size),
                class_to_idx=base_ds.class_to_idx,
            )
            val_ds = FolderClassificationDataset(
                roots=val_roots,
                transform=build_eval_transform(args.image_size),
                class_to_idx=base_ds.class_to_idx,
            )
    else:
        if mode == "hf_cache":
            full_train = HFCachedArrowClassificationDataset(
                roots=roots,
                transform=build_train_transform(args.image_size),
                split=args.hf_split,
                class_to_idx=base_ds.class_to_idx,
                image_column=args.hf_image_column,
                label_column=args.hf_label_column,
            )
            full_eval = HFCachedArrowClassificationDataset(
                roots=roots,
                transform=build_eval_transform(args.image_size),
                split=args.hf_split,
                class_to_idx=base_ds.class_to_idx,
                image_column=args.hf_image_column,
                label_column=args.hf_label_column,
            )
        else:
            full_train = FolderClassificationDataset(
                roots=roots,
                transform=build_train_transform(args.image_size),
                class_to_idx=base_ds.class_to_idx,
            )
            full_eval = FolderClassificationDataset(
                roots=roots,
                transform=build_eval_transform(args.image_size),
                class_to_idx=base_ds.class_to_idx,
            )

        n_total = len(full_train)
        val_size = int(n_total * args.val_ratio)
        val_size = max(1, val_size)
        train_size = n_total - val_size
        if train_size <= 0:
            raise RuntimeError(f"Dataset too small for val split: total={n_total}, val_size={val_size}")

        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n_total, generator=g).tolist()
        train_indices = perm[:train_size]
        val_indices = perm[train_size:]
        train_ds = Subset(full_train, train_indices)
        val_ds = Subset(full_eval, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(args.model_name, num_classes=num_classes, pretrained=True).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[INFO] DataParallel enabled on {torch.cuda.device_count()} GPUs", flush=True)
        model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_acc = -1.0
    best_path = os.path.join(args.output_dir, "best_vit_cls.pth")
    class_map_path = os.path.join(args.output_dir, "class_to_idx.pkl")
    with open(class_map_path, "wb") as f:
        pickle.dump(base_ds.class_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{args.epochs}", ncols=100)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * y.size(0)
            seen += int(y.size(0))
            pbar.set_postfix(loss=running_loss / max(seen, 1), lr=optimizer.param_groups[0]["lr"])

        val_loss, val_acc = evaluate(model, val_loader, device=device, use_bf16=use_bf16)
        scheduler.step()
        print(f"[EPOCH {epoch+1:03d}] val_loss={val_loss:.6f} val_acc={val_acc:.6f}", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            obj = {
                "model_state_dict": unwrap_model(model).state_dict(),
                "model_name": args.model_name,
                "num_classes": num_classes,
                "image_size": args.image_size,
                "class_to_idx_path": class_map_path,
                "best_val_acc": best_acc,
            }
            torch.save(obj, best_path)
            print(f"[SAVE] best checkpoint -> {best_path} (acc={best_acc:.6f})", flush=True)

    print(f"[DONE] training finished, best_val_acc={best_acc:.6f}", flush=True)


def extract(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    use_bf16 = bool(args.bf16) and device.type == "cuda"

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt["model_name"]
    num_classes = int(ckpt["num_classes"])

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    ds = FaceImagesDataset(face_dir=args.face_dir, transform=build_eval_transform(args.image_size))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    id_feats: Dict[str, List[np.ndarray]] = {}
    with torch.inference_mode():
        for x, paths in tqdm(loader, desc="Extract CLS", ncols=100):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                cls = extract_cls_features(model, x)
            arr = cls.float().cpu().numpy().astype(np.float32, copy=False)
            for i, p in enumerate(paths):
                rid = extract_id_from_path(p)
                if rid is None:
                    continue
                sid = normalize_id(rid, width=args.id_width)
                if sid not in id_feats:
                    id_feats[sid] = []
                id_feats[sid].append(arr[i : i + 1])

    os.makedirs(args.out_dir, exist_ok=True)
    saved = 0
    for sid in sorted(id_feats.keys()):
        out_path = os.path.join(args.out_dir, f"{sid}.pkl")
        feat = np.concatenate(id_feats[sid], axis=0).astype(np.float32, copy=False)
        with open(out_path, "wb") as f:
            pickle.dump(feat, f, protocol=pickle.HIGHEST_PROTOCOL)
        saved += 1
    print(f"[DONE] features saved -> {args.out_dir} | num_ids={saved}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="ViT training and [CLS] feature extraction.")
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_train = sub.add_parser("train", help="Train ViT classifier.")
    ap_train.add_argument(
        "--train_roots",
        type=str,
        required=True,
        help="Comma-separated dataset roots. Example: /data/affectnethq,/data/casia_webface",
    )
    ap_train.add_argument(
        "--val_roots",
        type=str,
        default="",
        help="Optional comma-separated validation roots. If empty, split from train by --val_ratio.",
    )
    ap_train.add_argument("--val_ratio", type=float, default=0.1)
    ap_train.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    ap_train.add_argument("--image_size", type=int, default=224)
    ap_train.add_argument("--batch_size", type=int, default=64)
    ap_train.add_argument("--epochs", type=int, default=20)
    ap_train.add_argument("--num_workers", type=int, default=8)
    ap_train.add_argument("--lr", type=float, default=3e-5)
    ap_train.add_argument("--weight_decay", type=float, default=0.05)
    ap_train.add_argument("--min_lr", type=float, default=1e-6)
    ap_train.add_argument("--seed", type=int, default=42)
    ap_train.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap_train.add_argument("--bf16", type=int, default=1)
    ap_train.add_argument("--output_dir", type=str, default="outputs/vit_cls")
    ap_train.add_argument("--train_format", type=str, default="auto", choices=["auto", "folder", "hf_cache"])
    ap_train.add_argument("--hf_split", type=str, default="train")
    ap_train.add_argument("--hf_image_column", type=str, default="")
    ap_train.add_argument("--hf_label_column", type=str, default="")

    ap_extract = sub.add_parser("extract", help="Extract CLS features and save per-id pkl files.")
    ap_extract.add_argument("--checkpoint", type=str, required=True, help="Path to best_vit_cls.pth")
    ap_extract.add_argument("--face_dir", type=str, required=True, help="Directory containing face images")
    ap_extract.add_argument("--out_dir", type=str, default="images_feats_EmoVit")
    ap_extract.add_argument("--image_size", type=int, default=224)
    ap_extract.add_argument("--batch_size", type=int, default=256)
    ap_extract.add_argument("--num_workers", type=int, default=8)
    ap_extract.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap_extract.add_argument("--bf16", type=int, default=1)
    ap_extract.add_argument("--id_width", type=int, default=5)

    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "extract":
        extract(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
