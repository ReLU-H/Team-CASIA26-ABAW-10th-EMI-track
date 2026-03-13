#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common utilities for EmoViT training and feature extraction."""

import datetime
import os
import random
import re
from io import BytesIO
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp() -> Tuple[bool, int, int, int]:
    """Initialize DDP from torchrun env, returns (is_ddp, rank, local_rank, world_size)."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    timeout_seconds = int(os.environ.get("DDP_TIMEOUT_SECONDS", "7200"))
    is_ddp = world_size > 1
    if is_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return is_ddp, rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean_tensor(x: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= float(world_size)
    return y


def list_dataset_columns(ds_split: Any) -> List[str]:
    names = []
    if hasattr(ds_split, "column_names"):
        names = list(ds_split.column_names)
    return names


def detect_image_column(ds_split: Any, override: str = "") -> str:
    if override:
        return override
    names = list_dataset_columns(ds_split)
    preferred = ["image", "img", "pixel_values", "face", "jpg", "png"]
    lowered = {n.lower(): n for n in names}
    for p in preferred:
        if p in lowered:
            return lowered[p]
    for n in names:
        ln = n.lower()
        if "image" in ln or "img" in ln:
            return n
    raise RuntimeError(f"Cannot detect image column from: {names}")


def detect_label_column(ds_split: Any, override: str = "") -> str:
    if override:
        return override
    names = list_dataset_columns(ds_split)
    preferred = ["label", "labels", "emotion", "expression", "category", "target"]
    lowered = {n.lower(): n for n in names}
    for p in preferred:
        if p in lowered:
            return lowered[p]
    for n in names:
        ln = n.lower()
        if "label" in ln or "emotion" in ln or "expression" in ln:
            return n
    raise RuntimeError(f"Cannot detect label column from: {names}")


def maybe_to_pil(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, np.ndarray):
        arr = image_obj
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if isinstance(image_obj, dict):
        if "bytes" in image_obj and image_obj["bytes"] is not None:
            return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
        if "path" in image_obj and image_obj["path"]:
            return Image.open(image_obj["path"]).convert("RGB")
    if isinstance(image_obj, str):
        return Image.open(image_obj).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_obj)}")


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
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


def normalize_id(raw: Any, width: int = 5) -> str:
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
    """Extract a stable numeric id from path. Prefer directory id, then filename id."""
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


def shard_items(items: Sequence[Any], shard_id: int, num_shards: int) -> List[Any]:
    if num_shards <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    eps = 1e-12
    f1s: List[float] = []
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = (2.0 * p * r) / (p + r + eps)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def print_once(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def ddp_barrier(world_size: int, local_rank: Optional[int] = None) -> None:
    if world_size > 1 and dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            lr = int(os.environ.get("LOCAL_RANK", "0")) if local_rank is None else int(local_rank)
            dist.barrier(device_ids=[lr])
        else:
            dist.barrier()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common utilities for EmoViT training and feature extraction."""

import os
import random
import re
import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp() -> Tuple[bool, int, int, int]:
    """Initialize DDP from torchrun env, returns (is_ddp, rank, local_rank, world_size)."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    timeout_seconds = int(os.environ.get("DDP_TIMEOUT_SECONDS", "7200"))
    is_ddp = world_size > 1
    if is_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return is_ddp, rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean_tensor(x: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= float(world_size)
    return y


def list_dataset_columns(ds_split: Any) -> List[str]:
    names = []
    if hasattr(ds_split, "column_names"):
        names = list(ds_split.column_names)
    return names


def detect_image_column(ds_split: Any, override: str = "") -> str:
    if override:
        return override
    names = list_dataset_columns(ds_split)
    preferred = ["image", "img", "pixel_values", "face", "jpg", "png"]
    lowered = {n.lower(): n for n in names}
    for p in preferred:
        if p in lowered:
            return lowered[p]
    for n in names:
        ln = n.lower()
        if "image" in ln or "img" in ln:
            return n
    raise RuntimeError(f"Cannot detect image column from: {names}")


def detect_label_column(ds_split: Any, override: str = "") -> str:
    if override:
        return override
    names = list_dataset_columns(ds_split)
    preferred = ["label", "labels", "emotion", "expression", "category", "target"]
    lowered = {n.lower(): n for n in names}
    for p in preferred:
        if p in lowered:
            return lowered[p]
    for n in names:
        ln = n.lower()
        if "label" in ln or "emotion" in ln or "expression" in ln:
            return n
    raise RuntimeError(f"Cannot detect label column from: {names}")


def maybe_to_pil(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, np.ndarray):
        arr = image_obj
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if isinstance(image_obj, dict):
        if "bytes" in image_obj and image_obj["bytes"] is not None:
            return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
        if "path" in image_obj and image_obj["path"]:
            return Image.open(image_obj["path"]).convert("RGB")
    if isinstance(image_obj, str):
        return Image.open(image_obj).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_obj)}")


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
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


def normalize_id(raw: Any, width: int = 5) -> str:
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
    """Extract a stable numeric id from path. Prefer directory id, then filename id."""
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


def shard_items(items: Sequence[Any], shard_id: int, num_shards: int) -> List[Any]:
    if num_shards <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    eps = 1e-12
    f1s: List[float] = []
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = (2.0 * p * r) / (p + r + eps)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def print_once(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def ddp_barrier(world_size: int, local_rank: Optional[int] = None) -> None:
    if world_size > 1 and dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            lr = int(os.environ.get("LOCAL_RANK", "0")) if local_rank is None else int(local_rank)
            dist.barrier(device_ids=[lr])
        else:
            dist.barrier()

